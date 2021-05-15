import logging

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

from gnn_embed.model import MyGNN

from utils.graph_utils import get_batch_shift

#####################################################
#                 BATCHED OPERATIONS                #
#####################################################
EPS = 2e-2

def batched_expand(emb, batch):
    unique = torch.flip(torch.unique(batch.cpu(), sorted=False).to(batch.device), 
                        dims=(0,)) # TODO (Yulun): temp fix due to torch.unique bug

    X = torch.repeat_interleave(emb, torch.bincount(batch)[unique], dim=0)
    return X

def batched_sample(probs, batch):
    unique = torch.flip(torch.unique(batch.cpu(), sorted=False).to(batch.device), 
                        dims=(0,)) # TODO (Yulun): temp fix due to torch.unique bug
    mask = batch.unsqueeze(0) == unique.unsqueeze(1)

    p = probs * mask
    m = Categorical(p * (p > EPS))
    a = m.sample()
    return a

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
#                       GAPN                        #
#####################################################

class ActorCriticGAPN(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 eta,
                 eps_clip,
                 input_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden,
                 enc_nb_output):
        super(ActorCriticGAPN, self).__init__()
        # actor
        self.actor = GAPN_Actor(eta,
                                eps_clip,
                                input_dim,
                                nb_edge_types,
                                gnn_nb_layers,
                                gnn_nb_hidden,
                                enc_nb_layers,
                                enc_nb_hidden,
                                enc_nb_output)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr[0], betas=betas, eps=eps)
        # critic
        self.critic = GAPN_Critic(input_dim,
                                  nb_edge_types,
                                  gnn_nb_layers,
                                  gnn_nb_hidden,
                                  enc_nb_layers,
                                  enc_nb_hidden)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr[1], betas=betas, eps=eps)

    def forward(self):
        raise NotImplementedError

    def select_action(self, states, candidates, batch_idx):
        return self.actor.select_action(states, candidates, batch_idx)

    def get_value(self, states):
        return self.critic.get_value(states)

    def update(self, old_states, old_candidates, old_actions, old_logprobs, old_values, rewards, batch_idx):
        # Update actor
        loss = self.actor.loss(old_states, old_candidates, old_actions, old_logprobs, old_values, rewards, batch_idx)

        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()

        # Update critic
        baseline_loss = self.critic.loss(old_states, rewards)

        self.optimizer_critic.zero_grad()
        baseline_loss.backward()
        self.optimizer_critic.step()

        return loss.item(), baseline_loss.item()


class GAPN_Critic(nn.Module):
    def __init__(self,
                 input_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden):
        super(GAPN_Critic, self).__init__()
        self.gnn = MyGNN(input_dim, gnn_nb_hidden, gnn_nb_layers, nb_edge_types)
        if gnn_nb_layers == 0:
            in_dim = input_dim
        else:
            in_dim = gnn_nb_hidden

        layers = []
        for _ in range(enc_nb_layers):
            layers.append(nn.Linear(in_dim, enc_nb_hidden))
            in_dim = enc_nb_hidden

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(in_dim, 1)
        self.act = nn.ReLU()

        self.MseLoss = nn.MSELoss()

    def forward(self, g_emb):
        X = self.gnn.get_embedding(g_emb, detach=False)
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)

    def get_value(self, g_emb):
        with torch.autograd.no_grad():
            values = self(g_emb)
        return values.detach()

    def loss(self, g_emb, rewards):
        values = self(g_emb)
        loss = self.MseLoss(values, rewards)

        return loss


class GAPN_Actor(nn.Module):
    def __init__(self,
                 eta,
                 eps_clip,
                 input_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden,
                 enc_nb_output):
        super(GAPN_Actor, self).__init__()
        self.eta = eta
        self.eps_clip = eps_clip
        self.d_k = enc_nb_output
        self.gnn = MyGNN(input_dim, gnn_nb_hidden, gnn_nb_layers, nb_edge_types)
        if gnn_nb_layers == 0:
            in_dim = input_dim
        else:
            in_dim = gnn_nb_hidden

        Q_layers = []
        K_layers = []
        for _ in range(enc_nb_layers):
            Q_layers.append(nn.Linear(in_dim, enc_nb_hidden))
            K_layers.append(nn.Linear(in_dim, enc_nb_hidden))
            in_dim = enc_nb_hidden

        self.Q_layers = nn.ModuleList(Q_layers)
        self.K_layers = nn.ModuleList(K_layers)
        self.Q_final_layer = nn.Linear(in_dim, enc_nb_output)
        self.K_final_layer = nn.Linear(in_dim, enc_nb_output)

        self.final_layer = nn.Linear(2*enc_nb_output, 1)
        self.act = nn.ReLU()

    def forward(self, g, g_candidates, batch_idx):
        Q = self.gnn.get_embedding(g, detach=False)
        K = self.gnn.get_embedding(g_candidates, detach=False)

        for ql, kl in zip(self.Q_layers, self.K_layers):
            Q = self.act(ql(Q))
            K = self.act(kl(K))
        Q = self.Q_final_layer(Q)
        K = self.K_final_layer(K)

        Q = batched_expand(Q, batch_idx)
        logits = torch.sum(Q * K, dim=1) / self.d_k**.5
        #logits = self.final_layer(torch.cat([Q, K], dim=-1)).squeeze(1)

        probs = batched_softmax(logits, batch_idx)
        shifted_actions = batched_sample(probs, batch_idx)
        actions = shifted_actions - get_batch_shift(batch_idx)
        action_logprobs = torch.log(probs[shifted_actions])

        # probs for eval
        return probs, action_logprobs, actions

    def select_action(self, g, g_candidates, batch_idx):
        _, action_logprobs, actions = self(g, g_candidates, batch_idx)

        action_logprobs = action_logprobs.squeeze_().tolist()
        actions = actions.squeeze_().tolist()

        return action_logprobs, actions

    def evaluate(self, g, g_candidates, actions, batch_idx):
        Q = self.gnn.get_embedding(g, detach=False)
        K = self.gnn.get_embedding(g_candidates, detach=False)
        for ql, kl in zip(self.Q_layers, self.K_layers):
            Q = self.act(ql(Q))
            K = self.act(kl(K))
        Q = self.Q_final_layer(Q)
        K = self.K_final_layer(K)

        Q = batched_expand(Q, batch_idx)
        logits = torch.sum(Q * K, dim=1) / self.d_k**.5
        #logits = self.final_layer(torch.cat([Q, K], dim=-1)).squeeze(1)

        probs = batched_softmax(logits, batch_idx)
        batch_shift = get_batch_shift(batch_idx)
        shifted_actions = actions + batch_shift
        return probs[shifted_actions]

    def loss(self, g, g_candidates, actions, old_logprobs, state_values, rewards, batch_idx):
        probs = self.evaluate(g, g_candidates, actions, batch_idx)
        logprobs = torch.log(probs)
        entropies = probs * logprobs

        # Finding the ratio (pi_theta / pi_theta_old):
        ratios = torch.exp(logprobs - old_logprobs)
        advantages = rewards - state_values

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2)
        if torch.isnan(loss).any():
            logging.info("found nan in loss")
            exit()

        loss += self.eta * entropies

        return loss.mean()

