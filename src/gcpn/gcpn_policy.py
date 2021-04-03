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

from .gnn import GNN_Embed
from .mlp import Action_Prediction, Value_Network

class ActorCriticGCPN(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(ActorCriticGCPN, self).__init__()

        self.actor = GCPN_Actor(input_dim,
                               emb_dim,
                               nb_edge_types,
                               gnn_nb_layers,
                               gnn_nb_hidden,
                               gnn_heads,
                               mlp_nb_layers,
                               mlp_nb_hidden)
        # critic
        self.critic = GCPN_Critic(input_dim,
                               emb_dim,
                               nb_edge_types,
                               gnn_nb_layers,
                               gnn_nb_hidden,
                               gnn_heads,
                               mlp_nb_layers,
                               mlp_nb_hidden)
        '''
        self.critic = Value_Network(emb_dim,
                                    mlp_nb_layers,
                                    mlp_nb_hidden)
        '''

    def forward(self):
        raise NotImplementedError

    def act(self, states, candidates, batch_idx):
        with torch.autograd.no_grad():
            probs, actions, shifted_actions = self.actor(states, candidates, batch_idx)

        action_logprobs = torch.log(probs).squeeze_().tolist()
        actions = actions.squeeze_().tolist()
        shifted_actions = shifted_actions.squeeze_().tolist()

        return action_logprobs, actions, shifted_actions

    def evaluate(self, states, candidates, batch_idx, actions):   
        probs = self.actor.evaluate(states, candidates, batch_idx, actions)

        action_logprobs = torch.log(probs)
        state_values = self.critic(states)
        '''
        state_values = self.critic(self.actor.gnn_embed(states))
        '''

        entropy = probs * action_logprobs

        return action_logprobs, state_value, entropy


class GCPN_Actor(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(GCPN_Actor, self).__init__()

        self.gnn_embed = GNN_Embed(gnn_nb_hidden,
                                   gnn_nb_layers,
                                   gnn_heads,
                                   input_dim,
                                   emb_dim,
                                   True)
        self.action_prediction = Action_Prediction(mlp_nb_layers,
                                                   mlp_nb_hidden,
                                                   2*emb_dim)

    def get_emb(self, g, g_candidates, batch_idx):
        g_emb = self.gnn_embed(g)
        g_candidates_emb = self.gnn_embed(g_candidates)

        X = torch.repeat_interleave(g_emb, torch.bincount(batch_idx), dim=0)
        X = torch.cat((X, g_candidates_emb), dim=1)
        return X

    def forward(self, g, g_candidates, batch_idx):
        X = self.get_emb(g, g_candidates, batch_idx)
        p, actions, shifted_actions = self.action_prediction(X, batch_idx)

        return p, actions, shifted_actions

    def evaluate(self, g, g_candidates, batch_idx, actions):
        X = self.get_emb(g, g_candidates, batch_idx)
        p = self.action_prediction.evaluate(X, batch_idx, actions)

        return p


class GCPN_Critic(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(GCPN_Critic, self).__init__()

        self.gnn_embed = GNN_Embed(gnn_nb_hidden,
                                   gnn_nb_layers,
                                   gnn_heads,
                                   input_dim,
                                   emb_dim,
                                   True)
        self.value_network = Value_Network(emb_dim, 
                                           mlp_nb_layers, 
                                           mlp_nb_hidden)

    def forward(self, g):
        X = self.gnn_embed(g)
        return self.value_network(X)

