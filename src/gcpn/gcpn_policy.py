import time
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree


#####################################################
#                   HELPER MODULES                  #
#####################################################
def batched_sample(probs, batch):
    unique = torch.unique(batch)
    mask = batch.unsqueeze(0) == unique.unsqueeze(1)

    m = Categorical(probs * mask)
    a = m.sample()
    return a, probs[a]

def batched_softmax(logits, batch):
    logits = torch.exp(logits)

    logit_sum = pyg.nn.global_add_pool(logits, batch)
    logit_sum = torch.index_select(logit_sum, 0, batch)
    probs = torch.div(logits, logit_sum)
    return probs

def get_batch_shift(pyg_batch):
    batch_num_nodes = torch.bincount(pyg_batch)

    # shift batch
    zero = torch.LongTensor([0]).to(batch_num_nodes.device)
    cumsum = torch.cat((zero, torch.cumsum(batch_num_nodes, dim=0)[:-1]))

    return cumsum

#####################################################
#                       CREM                        #
#####################################################

class GCPN_CReM(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(GCPN_CReM, self).__init__()

        layers = [nn.Linear(2*emb_dim, mlp_nb_hidden)]
        for _ in range(mlp_nb_layers-1):
            layers.append(nn.Linear(mlp_nb_hidden, mlp_nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(mlp_nb_hidden, 1)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(0)

    def forward(self, g, g_candidates, surrogate_model, batch_idx):
        g_emb = self.get_embedding(g, surrogate_model)
        g_candidates_emb = self.get_embedding(g_candidates, surrogate_model)

        X = torch.repeat_interleave(g_emb, torch.bincount(batch_idx), dim=0)
        X = torch.cat((X, g_candidates_emb), dim=1)
        X_states = X

        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        X = self.final_layer(X).squeeze(1)
        probs = batched_softmax(X, batch_idx)

        if self.training:
            a, p = batched_sample(probs, batch_idx)
            batch_shift = get_batch_shift(batch_idx)
            actions = a - batch_shift
            return g_emb, X_states, actions, p
        else:
            return g_emb, X_states, probs

    def get_embedding(self, g, surrogate_model):
        with torch.autograd.no_grad():
            emb = surrogate_model.get_embedding(g)
        return emb

    def evaluate(self, candidates, actions):
        emb = candidates.x
        for i, l in enumerate(self.layers):
            emb = self.act(l(emb))

        logits = self.final_layer(emb).squeeze(1)
        probs = batched_softmax(logits, candidates.batch)
        
        batch_shift = get_batch_shift(candidates.batch)
        shifted_actions = actions + batch_shift
        return probs[shifted_actions]

