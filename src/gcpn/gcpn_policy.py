import numpy as np
from random import choices

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_scatter import scatter_add

from utils.graph_utils import mol_to_pyg_graph
import copy

from rdkit import Chem
from crem.crem import mutate_mol


def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


class GCPN_crem(nn.Module):
    def __init__(self,
                 input_dim,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 emb_dim,
                 mlp_nb_layers,
                 mlp_nb_hidden,
                 sample_crem,
                 device):
        super(GCPN_crem, self).__init__()

        # TODO: Normalization of probabilities is inconsistent between possible crem states, and evaluating states.
        self.gnn_embed = GNN_Embed(gnn_nb_hidden,
                                   gnn_nb_layers,
                                   gnn_heads,
                                   input_dim,
                                   emb_dim,
                                   True)
        self.mc = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    emb_dim)
        self.emb_dim = emb_dim
        self.device = device
        self.sample_crem = sample_crem

    def forward(self, mol, eval_action=None):
        """Find's list of molecule mutations with CReM, then feeds them to a GNN_embedding network, then a MLP.
        Need to return action, prob, and list of states."""

        # Adhoc rdkit fixes for mol representation
        mol.UpdatePropertyCache(strict=False)
        mol = convert_radical_electrons_to_hydrogens(mol)
        Chem.SanitizeMol(mol,
                         Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE | \
                         Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | \
                         Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)

        # CReM
        db_fname = 'replacements02_sc2.db'

        try:
            new_mols = list(mutate_mol(mol, db_fname, return_mol=True))
            print("CReM options:" + str(len(new_mols)))
            new_mols = [Chem.RemoveHs(i[1]) for i in new_mols]
            if len(new_mols) > self.sample_crem:
                print("Downsampling to 20 options.")
                new_mols = choices(new_mols, k=self.sample_crem)
        except Exception as e:
            print(e)
            new_mols = []
        new_mols.append(mol)  # Also consider the molecule by itself, if chosen stop is implied.
        new_pygs = Batch().from_data_list([mol_to_pyg_graph(i) for i in new_mols]).to(self.device)

        with torch.autograd.no_grad():
            # Policy
            if len(new_mols) == 1:
                action, prob = -1, torch.tensor(1.0)
            else:
                X = self.gnn_embed(new_pygs)
                f_probs = self.mc(X)  # Mask is not needed since each row is a molecule.
                action, prob = sample_from_probs(f_probs, eval_action)

            if action == (len(new_mols) - 1):
                action = -1  # Token for stop.
        return action, prob, new_mols

    def get_crem_opt(self, X, eval_actions):
        f_probs = self.mc(X)
        probs = f_probs[eval_actions]
        return probs

    # TODO (Andrew): Need to change how things are evaluated
    def evaluate(self, orig_states, actions):
        # batches = [torch.tensor(step.batch) for step in orig_states]
        n_steps = len(orig_states)
        p_agg = torch.empty(n_steps).to(self.device)
        X_agg = torch.empty((n_steps, self.emb_dim)).to(self.device)
        for i, batch in enumerate(orig_states):
            X = self.gnn_embed(batch)  # (n_crem, 128)
            p_all = self.mc(X)
            if p_all.ndim == 0:
                # When there's only one molecule, need to unsqueeze so indexing works
                p_all = torch.unsqueeze(p_all, 0)
            p_crem = p_all[actions[i].item()]
            p_agg[i] = p_crem
            X_agg[i] = X.sum(0)
        return p_agg, X_agg


class GCPN(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(GCPN, self).__init__()

        self.gnn_embed = GNN_Embed(gnn_nb_hidden,
                                   gnn_nb_layers,
                                   gnn_heads,
                                   input_dim,
                                   emb_dim)
        self.mf = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    emb_dim)
        self.ms = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    2 * emb_dim)
        self.me = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    2 * emb_dim,
                                    nb_edge_types=nb_edge_types,
                                    apply_softmax=False)
        self.mt = Action_Prediction(mlp_nb_layers,
                                    mlp_nb_hidden,
                                    emb_dim,
                                    apply_softmax=False)

    def forward(self, graph):
        nb_nodes = graph.x.shape[0]
        mask = torch.ones(nb_nodes).to(graph.x.device)
        X = self.gnn_embed(graph)

        a_first, p_first = self.get_first(X, mask)
        a_second, p_second = self.get_second(X, a_first, mask)
        a_edge, p_edge = self.get_edge(X, a_first, a_second)
        a_stop, p_stop = self.get_stop(X)

        actions = np.array([[a_first, a_second, a_edge, a_stop]])
        probs = torch.stack((p_first, p_second, p_edge, p_stop))
        return actions, probs

    def get_first(self, X, mask=None, eval_action=None):
        f_probs = self.mf(X)
        if mask is not None:
            nb_true_nodes = int(sum(mask)) - 9
            true_node_mask = mask.clone()
            true_node_mask[nb_true_nodes:] = 0.0
            f_probs = f_probs * true_node_mask  # UNSURE, CHECK
        return sample_from_probs(f_probs, eval_action)

    def get_second(self, X, a_first, mask=None, eval_action=None):
        nb_nodes = X.shape[0]
        X_first = X[a_first].unsqueeze(0).repeat(nb_nodes, 1)
        X_cat = torch.cat((X_first, X), dim=1)

        f_probs = self.ms(X_cat)
        if mask is not None:
            f_probs = f_probs * mask  # UNSURE, CHECK
            f_probs[a_first] = 0.0
        return sample_from_probs(f_probs, eval_action)

    def get_edge(self, X, a_first, a_second, eval_action=None):
        X_first = X[a_first]
        X_second = X[a_second]
        X_cat = torch.cat((X_first, X_second), dim=0)

        f_logits = self.me(X_cat)
        f_probs = nn.functional.softmax(f_logits, dim=0)
        return sample_from_probs(f_probs, eval_action)

    def get_stop(self, X, eval_action=None):
        X_agg = X.mean(0)
        f_logit = self.mt(X_agg)
        f_prob = torch.sigmoid(f_logit).squeeze()

        m = Bernoulli(f_prob)
        a = (m.sample().item()) if eval_action is None else eval_action
        f_prob = 1 - f_prob if a == 0 else f_prob  # probability of choosing 0
        return int(a), f_prob

    def get_first_opt(self, X, eval_actions, batch):
        f_probs = self.mf(X, batch)
        probs = f_probs[eval_actions]
        return probs

    def get_second_opt(self, X, a_first, a_second, batch):
        X_first = X[a_first]
        X_first = torch.index_select(X_first, 0, batch)
        X_cat = torch.cat((X_first, X), dim=1)

        f_probs = self.ms(X_cat, batch)
        p_second = f_probs[a_second]
        return p_second

    def get_edge_opt(self, X, a_first, a_second, a_edge):
        X_first = X[a_first]
        X_second = X[a_second]
        X_cat = torch.cat((X_first, X_second), dim=1)

        f_logits = self.me(X_cat)
        f_prob = nn.functional.softmax(f_logits, dim=1)
        probs = torch.gather(f_prob, 1, a_edge.unsqueeze(1)).squeeze()
        return probs

    def get_stop_opt(self, X, batch, a_stop):
        X_agg = pyg.nn.global_mean_pool(X, batch)

        prob_stop = torch.sigmoid(self.mt(X_agg))
        prob_not_stop = 1 - prob_stop
        f_prob = torch.cat((prob_not_stop, prob_stop), dim=1)

        prob = torch.gather(f_prob, 1, a_stop.unsqueeze(1)).squeeze()
        return prob

    def evaluate(self, orig_states, actions):
        batch = orig_states.batch
        X = self.gnn_embed(orig_states)

        a_first, a_second, batch_num_nodes, = get_batch_idx(batch, actions)
        # Each of these is (2000,1)
        p_first_agg = self.get_first_opt(X, a_first, batch)
        p_second_agg = self.get_second_opt(X, a_first, a_second, batch)
        p_edge_agg = self.get_edge_opt(X, a_first, a_second, actions[:, 2])
        p_stop_agg = self.get_stop_opt(X, batch, actions[:, 3])
        probs_agg = torch.stack((p_first_agg,
                                 p_second_agg,
                                 p_edge_agg,
                                 p_stop_agg),
                                dim=1)
        X_agg = pyg.nn.global_add_pool(X, batch)
        return probs_agg, X_agg


def get_batch_idx(batch, actions):
    """Finds row index of X according to what action was taken at each time step"""
    batch_num_nodes = torch.bincount(batch)
    cumsum = torch.cumsum(batch_num_nodes, dim=0) - batch_num_nodes[0]

    a_first = cumsum + actions[:, 0]
    a_second = cumsum + actions[:, 1]
    return a_first, a_second, batch_num_nodes


def sample_from_probs(p, action):
    m = Categorical(p)
    a = m.sample() if action is None else action
    return a.item(), p[a]


class GNN_Embed(nn.Module):
    def __init__(self,
                 nb_hidden_gnn,
                 nb_layer,
                 heads,
                 input_dim,
                 emb_dim,
                 crem=False):
        super(GNN_Embed, self).__init__()

        gnn_layers = [MyGCNConv(input_dim,
                                nb_hidden_gnn,
                                use_attention=True,
                                heads=heads,
                                nb_edge_attr=1)]
        for _ in range(nb_layer - 1):
            gnn_layers.append(MyGCNConv(nb_hidden_gnn,
                                        nb_hidden_gnn,
                                        batch_norm=True,
                                        use_attention=True,
                                        heads=heads,
                                        nb_edge_attr=1))

        self.layers = nn.ModuleList(gnn_layers)
        self.final_emb = nn.Linear(nb_hidden_gnn, emb_dim)
        self.crem = crem

    def forward(self, data):

        emb = data.x
        # GNN Layers
        for i, layer in enumerate(self.layers):
            emb = layer(emb, data.edge_index, data.edge_attr)

        emb = self.final_emb(emb)

        # If using crem, we need to pool node representations.
        if self.crem:
            emb = pyg.nn.global_mean_pool(emb, data.batch).squeeze()
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
        for _ in range(nb_layers - 1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, nb_edge_types)
        self.act = nn.ReLU()
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax(dim=0)

    def forward(self, X, batch=None):
        for l in self.layers:
            X = self.act(l(X))
        logits = self.final_layer(X)

        if self.apply_softmax:
            logits = logits.squeeze()
            if batch is not None:
                probs = batched_softmax(logits, batch)
            else:
                probs = self.softmax(logits)
            return probs
        else:
            return logits


def batched_softmax(logits, batch):
    logits = torch.exp(logits)

    logit_sum = pyg.nn.global_add_pool(logits, batch)
    logit_sum = torch.index_select(logit_sum, 0, batch)
    probs = torch.div(logits, logit_sum)
    return probs


#####################################################
#                   HELPER MODULES                  #
#####################################################

class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, batch_norm=False, res=True,
                 use_attention=True, heads=1, nb_edge_attr=1, concat=False, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(MyGCNConv, self).__init__(aggr='add', node_dim=0, **kwargs)  # "Add" aggregation.
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


class MyHGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, batch_norm=False, res=True, norm_mode="symmetric",
                 use_attention=False, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        super(MyHGCN, self).__init__(aggr='add', node_dim=0, **kwargs)
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


from torch.nn.modules.instancenorm import _InstanceNorm
from .MLP import MyBatchNorm


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
