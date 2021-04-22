import torch
import torch.nn as nn

import torch_geometric as pyg


class GNN_MyGAT(nn.Module):
    def __init__(self, input_dim, emb_dim, nb_hidden, nb_layer, use_3d=False):
        super(GNN_MyGAT, self).__init__()
        if nb_layer == 1:
            layers = [MyGATConv(input_dim, emb_dim)]
        else:
            layers = [MyGATConv(input_dim, nb_hidden)]
            for _ in range(nb_layer-2):
                layers.append(MyGATConv(nb_hidden, nb_hidden))
            layers.append(MyGATConv(nb_hidden, emb_dim))
        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

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
            # if i==2:
            #     x = nn.functional.dropout(x, training=self.training)
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
    def __init__(self, in_channels, out_channels, batch_norm=False, res=True,
                 use_attention=True, heads=1, nb_edge_attr=1, concat=False, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(MyGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.res = res
        self.use_attention = use_attention
        self.nb_edge_attr = nb_edge_attr

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.linN = Parameter(
                torch.randn(in_channels, heads * out_channels) * (heads * out_channels) ** (-0.5))
            self.linE = Parameter(
                torch.randn(nb_edge_attr, heads * nb_edge_attr) * (heads * nb_edge_attr) ** (-0.5))
            self.att = Parameter(torch.randn(
                1, heads, 2 * out_channels + nb_edge_attr) * (2 * out_channels + nb_edge_attr) ** (-0.5))
        else:
            self.heads = 1
            self.concat = True
            self.linN = Parameter(torch.randn(in_channels, out_channels) * out_channels ** (-0.5))
            self.register_parameter('linE', None)

        if self.batch_norm:
            # self.norm = MyInstanceNorm(in_channels, track_running_stats=False)
            self.norm = MyBatchNorm(in_channels)

        if bias and concat:
            self.bias = Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.act = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linN)
        glorot(self.linE)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        if self.batch_norm:
            x = self.norm(x)

        if size is None and torch.is_tensor(x):
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index,
                                                   edge_weight=edge_attr,
                                                   fill_value=0.0,
                                                   num_nodes=x.size(self.node_dim))
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = edge_attr.view(-1, self.nb_edge_attr)

        x = torch.matmul(x, self.linN)
        alpha = None

        x = x.view(-1, self.heads, self.out_channels)
        if self.use_attention:
            edge_attr = torch.matmul(edge_attr, self.linE)
            edge_attr = edge_attr.view(-1, self.heads, self.nb_edge_attr)

            x_i, x_j = x[edge_index[0]], x[edge_index[1]]
            alpha = (torch.cat([x_i, x_j, edge_attr], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.propagate(edge_index, x=x, alpha=alpha)

        if self.concat is True:
            x = x.view(-1, self.heads * self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            x = x.mean(dim=1)  # TODO(Yulun): simply extract one entry of dim 1.
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if self.res:
            out += x

        out = self.act(out)
        return out

    def message(self, x_j, alpha):
        out = x_j
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MyHGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, batch_norm=False, res=True, norm_mode="symmetric",
                 use_attention=False, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(MyHGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.res = res
        self.norm_mode = norm_mode
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.randn(in_channels, heads * out_channels) * (heads * out_channels) ** (-0.5))
            self.att = Parameter(torch.randn(1, heads, 2 * out_channels) * (2 * out_channels) ** (-0.5))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.randn(in_channels, out_channels) * out_channels ** (-0.5))

        if self.batch_norm:
            # self.norm = MyInstanceNorm(in_channels, track_running_stats=False)
            self.norm = MyBatchNorm(in_channels)

        if bias and concat:
            self.bias = Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.act = nn.ReLU()

        # TODO(Yulun): reset params

    def forward(self, x, hyperedge_index, hyperedge_weight=None):

        if self.batch_norm:
            x = self.norm(x)

        x = torch.matmul(x, self.weight)
        alpha = None

        x = x.view(-1, self.heads, self.out_channels)
        if self.use_attention:
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        if self.norm_mode is "row":
            self.flow = 'source_to_target'
            out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        elif self.norm_mode is "col":
            self.flow = 'source_to_target'
            out = self.propagate(hyperedge_index, x=D.view(-1, 1, 1) * x, norm=B, alpha=alpha)
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=out, alpha=alpha)
        else:
            self.flow = 'source_to_target'
            out = self.propagate(hyperedge_index, x=D.pow(0.5).view(-1, 1, 1) * x, norm=B, alpha=alpha)
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=out, norm=D.pow(0.5), alpha=alpha)

        if self.concat is True:
            x = x.view(-1, self.heads * self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            x = x.mean(dim=1)  # TODO(Yulun): simply extract one entry of dim 1.
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if self.res:
            out += x

        out = self.act(out)
        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

