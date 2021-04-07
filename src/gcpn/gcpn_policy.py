import time
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

from utils.graph_utils import get_batch_shift

#####################################################
#                   HELPER MODULES                  #
#####################################################
EPS = 1e-4

def batched_sample(probs, batch):
    unique = torch.flip(torch.unique(batch.cpu(), sorted=False).to(batch.device), dims=(0,)) # temp fix due to torch.unique bug
    mask = batch.unsqueeze(0) == unique.unsqueeze(1)

    p = probs * mask
    m = Categorical(p * (p > EPS))
    a = m.sample()
    return a, probs[a]

def batched_softmax(logits, batch):
    logit_max = pyg.nn.global_max_pool(logits, batch)
    logit_max = torch.index_select(logit_max, 0, batch)

    logits = logits - logit_max
    logits = torch.exp(logits)

    logit_sum = pyg.nn.global_add_pool(logits, batch) + EPS
    logit_sum = torch.index_select(logit_sum, 0, batch)
    probs = torch.div(logits, logit_sum)
    return probs

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

    def get_embedding(self, g, surrogate_model):
        return surrogate_model.get_embedding(g).detach()

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

        shifted_actions, p = batched_sample(probs, batch_idx)
        actions = shifted_actions - get_batch_shift(batch_idx)
        action_logprobs = torch.log(p)
        return g_emb, X_states, probs, action_logprobs, actions, shifted_actions

    def select_action(self, g, g_candidates, surrogate_model, batch_idx):
        g_emb, X_states, probs, action_logprobs, actions, shifted_actions = self(g, g_candidates, surrogate_model, batch_idx)

        g_emb = g_emb.detach().cpu()
        X_states = X_states.detach().cpu()

        probs = probs.squeeze_().tolist()
        action_logprobs = action_logprobs.squeeze_().tolist()
        actions = actions.squeeze_().tolist()
        shifted_actions = shifted_actions.squeeze_().tolist()

        if self.training:
            return g_emb, X_states, action_logprobs, actions, shifted_actions
        else:
            return g_emb, X_states, probs

    def evaluate(self, candidates, actions):
        emb = candidates.x
        for i, l in enumerate(self.layers):
            emb = self.act(l(emb))

        logits = self.final_layer(emb).squeeze(1)
        probs = batched_softmax(logits, candidates.batch)

        batch_shift = get_batch_shift(candidates.batch)
        shifted_actions = actions + batch_shift
        return probs[shifted_actions]

