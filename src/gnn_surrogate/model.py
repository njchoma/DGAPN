import torch
import torch.nn as nn

import torch_geometric as pyg

class GNN(nn.Module):
    def __init__(self, input_dim, nb_hidden, nb_layer):
        super(GNN, self).__init__()
        layers = [pyg.nn.GATConv(input_dim, nb_hidden)]
        for _ in range(nb_layer-1):
            layers.append(pyg.nn.GATConv(nb_hidden, nb_hidden))
        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, g, dense_edge_idx):
        x = g.x
        edge_index = g.edge_index
        for l in self.layers:
            x = l(x, edge_index)
            x = self.act(x)
        x = pyg.nn.global_add_pool(x, g.batch)
        y = self.final_layer(x).squeeze()
        return y

class GNN_Dense(nn.Module):
    def __init__(self, input_dim, nb_hidden, nb_layer):
        super(GNN_Dense, self).__init__()
        assert (nb_hidden%2)==0

        layersA = [pyg.nn.GATConv(input_dim, nb_hidden//2)]
        for _ in range(nb_layer-1):
            layersA.append(pyg.nn.GATConv(nb_hidden, nb_hidden//2))
        self.layersA = nn.ModuleList(layersA)

        layersB = [pyg.nn.GATConv(input_dim, nb_hidden//2)]
        for _ in range(nb_layer-1):
            layersB.append(pyg.nn.GATConv(nb_hidden, nb_hidden//2))
        self.layersB = nn.ModuleList(layersA)

        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        for la, lb in zip(self.layersA, self.layersB):
            x_A = la(x, edge_index)
            x_B = lb(x, dense_edge_idx)
            x = torch.cat((x_A, x_B), dim=1)
            x = self.act(x)
        x = pyg.nn.global_add_pool(x, g.batch)
        y = self.final_layer(x).squeeze()
        return y





class GNN_MyGAT(nn.Module):
    def __init__(self, input_dim, nb_hidden, nb_layer, use_3d=False):
        super(GNN_MyGAT, self).__init__()
        layers = [MyGATConv(input_dim, nb_hidden)]
        for _ in range(nb_layer-1):
            layers.append(MyGATConv(nb_hidden, nb_hidden))
        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()
        self.emb_dim = nb_hidden

    def forward(self, g, dense_edge_idx=None):
        x = self.get_embedding(g)
        #x = torch.sum(x, dim=0)
        y = self.final_layer(x).squeeze()
        return y

    def get_embedding(self, g):
        x = g.x
        edge_index = g.edge_index
        edge_attr  = g.edge_attr
        for i, l in enumerate(self.layers):
            x = l(x, edge_index, edge_attr)
            x = self.act(x)
        x = pyg.nn.global_add_pool(x, g.batch)
        return x



import math
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class MyGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nb_edge_feats=1, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(MyGATConv, self).__init__(aggr='add', **kwargs)

        # assert heads==1 # does NOT work with more than one head as of pytorch_geometric 1.6.1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels+nb_edge_feats))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, edge_attr, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index,
                                           edge_weight=edge_attr,
                                           fill_value=0.0,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, edge_attr=edge_attr, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            edge_attr = edge_attr.view(-1, self.heads, 1)
            edge_feats = torch.cat([x_i, x_j, edge_attr], dim=-1)
            alpha = (edge_feats * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, edge_index_i, size_i)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        attended = x_j * alpha.view(-1, self.heads, 1)

        # NOTE: this does NOT work with multiple heads.
        # This is a quick hack to work with pytorch_geometric 1.6.1
        return attended.squeeze(1)
        # return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
