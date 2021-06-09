import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import Data, Batch

#####################################################
#                    MODEL SAVING                   #
#####################################################
def load_sGAT(state_path):
    state = torch.load(state_path)
    net = sGAT(state['input_dim'],
                state['nb_hidden'],
                state['nb_layers'],
                state['nb_edge_types'],
                state['use_3d'])
    net.load_state_dict(state['state_dict'])
    return net

def save_sGAT(net, state_path):
    torch.save(net.get_dict(), state_path)

#####################################################
#                        sGAT                       #
#####################################################

class sGAT(nn.Module):
    def __init__(self, input_dim, nb_hidden, nb_layers, nb_edge_types, use_3d=False, init_method='uniform'):
        super(sGAT, self).__init__()
        self.input_dim = input_dim
        self.nb_hidden = nb_hidden
        self.nb_layers = nb_layers
        self.nb_edge_types = nb_edge_types
        self.use_3d = use_3d

        layers = []
        in_dim = input_dim
        for _ in range(nb_layers):
            layers.append(MyGAT(in_dim, nb_hidden, nb_edge_types, init_method=init_method))
            in_dim = nb_hidden
        self.layers = nn.ModuleList(layers)

        if use_3d:
            layers3D = []
            in_dim = input_dim
            for _ in range(nb_layers):
                layers3D.append(MyGCN(in_dim, nb_hidden, init_method=init_method))
                in_dim = nb_hidden
            self.layers3D = nn.ModuleList(layers3D)

        self.final_layer = nn.Linear(in_dim, 1)

    def forward(self, g, g3D=None):
        x = self.get_embedding(g, g3D, detach=False)
        y = self.final_layer(x).squeeze()
        return y

    def get_embedding(self, g, g3D=None, n_layers=None, return_3d=False, aggr=True, detach=True):
        if isinstance(g, list):
            g3D = g[1]
            g = g[0]
        if n_layers is None:
            n_layers = self.nb_layers
        assert n_layers <= self.nb_layers

        x = g.x
        edge_index = g.edge_index
        edge_attr  = g.edge_attr
        batch = g.batch
        if g3D is None:
            assert self.use_3d is False
            for i in range(n_layers):
                x, edge_attr = self.layers[i](x, edge_index, edge_attr)
        else:
            geom_index = g3D.edge_index
            geom_attr = g3D.edge_attr
            for i in range(n_layers):
                x1, edge_attr = self.layers[i](x, edge_index, edge_attr)
                x2 = self.layers3D[i](x, geom_index, geom_attr)
                x = x1 + x2

        if detach:
            x = x.detach()
            edge_attr = edge_attr.detach()

        if aggr:
            return pyg.nn.global_add_pool(x, batch)
        else:
            g.x = x
            g.edge_attr = edge_attr
            if return_3d:
                g3D.x = x
                return [g, g3D]
            else:
                return g

    def to_device(self, device, n_layers=None):
        if n_layers is None:
            n_layers = self.nb_layers
        assert n_layers <= self.nb_layers

        for i in range(n_layers):
            self.layers[i].to(device)
            if self.use_3d:
                self.layers3D[i].to(device)

    def get_dict(self):
        state = {'state_dict': self.state_dict(),
                    'input_dim': self.input_dim,
                    'nb_hidden': self.nb_hidden,
                    'nb_layers': self.nb_layers,
                    'nb_edge_types': self.nb_edge_types,
                    'use_3d': self.use_3d}
        return state



import math
from torch.nn import Parameter, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_scatter import scatter_add

def init_randoms(tensor, method='uniform'):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        if method == 'uniform':
            tensor.data.uniform_(-stdv, stdv)
        elif method == 'normal':
            tensor.data.normal_(-stdv, stdv)
        else:
            pass

def init_zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class MyGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, nb_edge_attr, batch_norm=False, res=False,
                 heads=2, concat=False, negative_slope=0.2, dropout=0, bias=True, init_method='uniform',
                 **kwargs):
        super(MyGAT, self).__init__(aggr='add', node_dim=0, **kwargs)  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.res = res
        self.nb_edge_attr = nb_edge_attr

        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.linN = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.linE = Parameter(torch.Tensor(nb_edge_attr, heads * nb_edge_attr))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + nb_edge_attr))

        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        if self.batch_norm:
            # self.norm = MyInstanceNorm(in_channels, track_running_stats=False)
            self.norm = BatchNorm1d(in_channels)

        self.act = nn.ReLU()

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.linN, init_method)
        init_randoms(self.linE, init_method)
        init_randoms(self.att, init_method)
        init_zeros(self.bias)

    def forward(self, node_attr, edge_index, edge_attr):
        # node_attr has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, nb_edge_attr]
        x = node_attr

        if self.batch_norm:
            x = self.norm(x)

        x = torch.matmul(x, self.linN)
        if self.bias is not None:
            x += self.bias
        x = x.view(-1, self.heads, self.out_channels)

        edge_attr = edge_attr.view(-1, self.nb_edge_attr)
        edge_attr = torch.matmul(edge_attr, self.linE)
        edge_attr = edge_attr.view(-1, self.heads, self.nb_edge_attr)

        x_i, x_j = x[edge_index[0]], x[edge_index[1]]
        alpha = (torch.cat([x_i, x_j, edge_attr], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.propagate(edge_index, x=x, alpha=alpha)
        out += x

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_attr = edge_attr.view(-1, self.heads * self.nb_edge_attr)
        else:
            # TODO (Yulun): Efficiency
            out = out.mean(dim=1)
            edge_attr = edge_attr.mean(dim=1)

        out = self.act(out)
        edge_attr = self.act(edge_attr)

        if self.res:
            out += node_attr
        return out, edge_attr

    def message(self, x_j, alpha):
        return alpha.view(-1, self.heads, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels, self.out_channels, self.heads)


class MyGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, batch_norm=False, res=False, 
                 dropout=0, bias=True, init_method='uniform', **kwargs):
        super(MyGCN, self).__init__(aggr='add', node_dim=0, **kwargs)  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.res = res

        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.batch_norm:
            # self.norm = MyInstanceNorm(in_channels, track_running_stats=False)
            self.norm = BatchNorm1d(in_channels)

        self.act = nn.ReLU()

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.weight, init_method)
        init_zeros(self.bias)

    def forward(self, node_attr, edge_index, edge_weight=None):
        # node_attr has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_weight has shape [E]
        if edge_weight is None:
            edge_weight = node_attr.new_ones(edge_index.size(1))
        x = node_attr

        if self.batch_norm:
            x = self.norm(x)

        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias

        D = scatter_add(edge_weight[edge_index[1]],
                    edge_index[0], dim=0, dim_size=x.size(0))
        D = D.pow(-0.5)
        D[D == float("inf")] = 0

        out = self.propagate(edge_index, x=D.view(-1,1)*x, norm=D)
        out += x

        out = self.act(out)

        if self.res:
            out += node_attr
        return out

    def message(self, x_j, edge_index_i, norm=None):
        out = x_j
        if norm is not None:
            norm_i = F.dropout(norm[edge_index_i], p=self.dropout, training=self.training)
            out = norm_i.view(-1, 1) * out
        return out

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                             self.in_channels, self.out_channels)


class MyHGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nb_edge_attr, batch_norm=False, res=True, norm_mode="symmetric",
                 heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, init_method='uniform',
                 **kwargs):
        super(MyHGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.res = res
        self.norm_mode = norm_mode
        self.nb_edge_attr = nb_edge_attr

        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.linN = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.linE = Parameter(torch.Tensor(nb_edge_attr, heads * nb_edge_attr))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + nb_edge_attr))

        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        if self.batch_norm:
            # self.norm = MyInstanceNorm(in_channels, track_running_stats=False)
            self.norm = BatchNorm1d(in_channels)

        self.act = nn.ReLU()

        self.reset_parameters(init_method)

    def reset_parameters(self, init_method='uniform'):
        init_randoms(self.linN, init_method)
        init_randoms(self.linE, init_method)
        init_randoms(self.att, init_method)
        init_zeros(self.bias)

    def forward(self, node_attr, hyperedge_index, edge_attr):
        x = node_attr

        if self.batch_norm:
            x = self.norm(x)

        x = torch.matmul(x, self.linN)
        if self.bias is not None:
            x += self.bias
        x = x.view(-1, self.heads, self.out_channels)

        edge_attr = edge_attr.view(-1, self.nb_edge_attr)
        edge_attr = torch.matmul(edge_attr, self.linE)
        edge_attr = edge_attr.view(-1, self.heads, self.nb_edge_attr)

        x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j, edge_attr], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        B = B.view(-1, self.heads) * alpha

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, alpha=alpha, norm=B)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, alpha=alpha)

        out += x

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_attr = edge_attr.view(-1, self.heads * self.nb_edge_attr)
        else:
            out = out.mean(dim=1)  # TODO(Yulun): simply extract one entry of dim 1.
            edge_attr = edge_attr.mean(dim=1)

        out = self.act(out)
        edge_attr = self.act(edge_attr)

        if self.res:
            out += node_attr
        return out, edge_attr

    def message(self, x_j, edge_index_i, alpha, norm=None):
        out = x_j
        if norm is not None:
            out = norm[edge_index_i].view(-1, self.heads, 1) * out
        return alpha.view(-1, self.heads, 1) * out



from torch.nn.modules.instancenorm import _InstanceNorm


class MyInstanceNorm(_InstanceNorm):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super(MyInstanceNorm, self).__init__(in_channels, eps, momentum, affine,
                                             track_running_stats)

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        if self.training or not self.track_running_stats:
            count = degree(batch, batch_size, dtype=x.dtype).view(-1, 1)
            tmp = scatter_add(x, batch, dim=0, dim_size=batch_size)
            mean = tmp / count.clamp(min=1)

            tmp = (x - mean[batch])
            tmp = scatter_add(tmp * tmp, batch, dim=0, dim_size=batch_size)
            var = tmp / count.clamp(min=1)
            unbiased_var = tmp / (count - 1).clamp(min=1)

        if self.training and self.track_running_stats:
            momentum = self.momentum
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean.mean(dim=0)
            self.running_var = (1 - momentum) * self.running_var + momentum * unbiased_var.mean(dim=0)

        if not self.training and self.track_running_stats:
            mean = self.running_mean.view(1, -1).expand(batch_size, -1)
            var = self.running_var.view(1, -1).expand(batch_size, -1)

        out = (x - mean[batch]) / torch.sqrt(var[batch] + self.eps)

        if self.affine:
            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)

        return out