import numpy as np

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree


class GCPN(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_nb_hidden_kernel,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(GCPN, self).__init__()

        self.gnn_embed = GNN_Embed(gnn_nb_hidden,
                                   gnn_nb_layers,
                                   gnn_nb_hidden_kernel,
                                   1,
                                   input_dim,
                                   emb_dim)

        self.mf = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    emb_dim)
        self.ms = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    2*emb_dim)
        self.me = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    2*emb_dim,
                                    nb_edge_types=nb_edge_types)
        self.mt = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    emb_dim,
                                    apply_softmax=False)

    def forward(self, graph):
        mask = graph.x.sum(1)
        X = self.gnn_embed(graph)

        a_first,  p_first  = self.get_first(X, mask)
        a_second, p_second = self.get_second(X, a_first, mask)
        a_edge,   p_edge   = self.get_edge(X, a_first, a_second)
        a_stop,   p_stop   = self.get_stop(X)

        actions = np.array([[a_first, a_second, a_edge, a_stop]])
        probs = torch.stack((p_first, p_second, p_edge, p_stop))
        return actions, probs

    def get_first(self, X, mask=None, eval_action=None):
        f_probs = self.mf(X)
        if mask is not None:
            nb_true_nodes = int(sum(mask))-9
            true_node_mask = mask.clone()
            true_node_mask[nb_true_nodes:] = 0.0
            f_probs = f_probs*true_node_mask # UNSURE, CHECK
        return sample_from_probs(f_probs, eval_action)

    def get_second(self, X, a_first, mask=None, eval_action=None):
        nb_nodes = X.shape[0]
        X_first = X[a_first].unsqueeze(0).repeat(nb_nodes, 1)
        X_cat = torch.cat((X_first, X),dim=1)

        f_probs = self.ms(X_cat)
        if mask is not None:
            f_probs = f_probs*mask # UNSURE, CHECK
            f_probs[a_first] = 0.0
        return sample_from_probs(f_probs, eval_action)

    def get_edge(self, X, a_first, a_second, eval_action=None):
        X_first  = X[a_first]
        X_second = X[a_second]
        X_cat = torch.cat((X_first, X_second),dim=0)

        f_probs = self.me(X_cat)
        return sample_from_probs(f_probs, eval_action)

    def get_stop(self, X, eval_action=None):
        X_agg = X.mean(0)
        f_logit = self.mt(X_agg)
        f_prob = torch.sigmoid(f_logit).squeeze()

        m = Bernoulli(f_prob)
        a = (m.sample().item()) if eval_action is None else eval_action
        f_prob = 1-f_prob if a==0 else f_prob # probability of choosing 0
        return int(a), f_prob
        

    def evaluate(self, orig_states, actions):
        batch = orig_states.batch
        X = self.gnn_embed(orig_states)
        states = orig_states.clone()
        states.x = X
        states = states.to_data_list()
        probs = []
        for s, a in zip(states, actions):
            _, p_first  = self.get_first(s.x, eval_action=a[0])
            _, p_second = self.get_second(s.x, a[0], eval_action=a[1])
            _, p_edge   = self.get_edge(s.x, a[0], a[1], eval_action=a[2])
            _, p_stop   = self.get_stop(s.x, eval_action=a[3])
            probs.append(torch.stack([p_first, p_second, p_edge, p_stop]))
        probs = torch.stack(probs)
        X_agg = pyg.nn.global_add_pool(X, batch)
        return probs, X_agg



class GNN_Embed(nn.Module):
  def __init__(self,
               nb_hidden_gnn,
               nb_layer,
               nb_hidden_kernel,
               nb_kernel,
               input_dim,
               emb_dim):
    super(GNN_Embed, self).__init__()


    gnn_layers = [MyGCNConv(input_dim,
                            nb_hidden_gnn,
                            nb_hidden_kernel)]
    for _ in range(nb_layer-1):
        gnn_layers.append(MyGCNConv(nb_hidden_gnn,
                                    nb_hidden_gnn,
                                    nb_hidden_kernel,
                                    apply_norm=True))

    self.layers = nn.ModuleList(gnn_layers)
    self.final_emb = nn.Linear(nb_hidden_gnn, emb_dim)

  def forward(self, data):

    emb = data.x
    # GNN Layers
    for i, layer in enumerate(self.layers):
      emb = layer(emb, data.edge_index)

    emb = self.final_emb(emb)
    return emb


class Action_Prediction(nn.Module):
    def __init__(self, 
                 nb_layers,
                 nb_hidden,
                 input_dim,
                 nb_edge_types=1,
                 apply_softmax=True):
        super(Action_Prediction, self).__init__()

        layers = [nn.Linear(input_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, nb_edge_types)
        self.act = nn.ReLU()
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax(dim=0)

    def forward(self, X):
        for l in self.layers:
            X = self.act(l(X))
        logits = self.final_layer(X)

        if self.apply_softmax:
            logits = logits.squeeze()
            probs = self.softmax(logits)
            return probs
        else:
            return logits




#####################################################
#                   HELPER MODULES                  #
#####################################################

class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nb_hidden_kernel=0, apply_norm=False):
        super(MyGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(2*in_channels, out_channels)
        self.act = nn.ReLU()

        if nb_hidden_kernel>0:
            self.linA = torch.nn.Linear(in_channels*2, nb_hidden_kernel)
            self.linB = torch.nn.Linear(nb_hidden_kernel, 1)
            self.sigmoid = nn.Sigmoid()

        self.apply_norm = apply_norm
        if apply_norm:
            # self.norm = MyInstanceNorm(in_channels, track_running_stats=False)
            self.norm = MyBatchNorm(in_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        if self.apply_norm:
            x = self.norm(x)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_prop = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        x_new = self.act(self.lin(torch.cat([x,x_prop],dim=1)))
        return x_new

    def message(self, x_i, x_j):
        h = self.act(self.linA(torch.cat([x_i, x_j], dim=1)))
        w = self.sigmoid(self.linB(h))

        return w * x_j

    def update(self, aggr_out):
        return aggr_out




from torch.nn.modules.instancenorm import _InstanceNorm
from torch_scatter import scatter_add
from torch_geometric.utils import degree

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
            self.running_mean = (
                1 - momentum) * self.running_mean + momentum * mean.mean(dim=0)
            self.running_var = (
                1 - momentum
            ) * self.running_var + momentum * unbiased_var.mean(dim=0)

        if not self.training and self.track_running_stats:
            mean = self.running_mean.view(1, -1).expand(batch_size, -1)
            var = self.running_var.view(1, -1).expand(batch_size, -1)

        out = (x - mean[batch]) / torch.sqrt(var[batch] + self.eps)

        if self.affine:
            out = out * self.weight.view(1, -1) + self.bias.view(1, -1)

        return out

from torch.nn import BatchNorm1d


class MyBatchNorm(BatchNorm1d):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__(in_channels, eps, momentum, affine,
                                        track_running_stats)

    def forward(self, x):
        return super(MyBatchNorm, self).forward(x)


    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)

def sample_from_probs(p, action):
    m = Categorical(p)
    a = m.sample() if action is None else action
    return a.item(), p[a]
