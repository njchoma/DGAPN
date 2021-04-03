import torch
import torch.nn as nn

import torch_geometric as pyg
from torch.distributions.categorical import Categorical

from utils.graph_utils import get_batch_shift

#####################################################
#                   HELPER MODULES                  #
#####################################################
def batched_sample(probs, batch):
    unique = torch.flip(torch.unique(batch.cpu(), sorted=False).to(batch.device), dims=(0,)) # temp fix due to torch.unique bug
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

#####################################################

class Value_Network(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(Value_Network, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(nb_hidden, 1)

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(Discriminator, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_act(self.final_layer(X)).squeeze(1)


class Action_Prediction(nn.Module):
    def __init__(self,
                 nb_layers,
                 nb_hidden,
                 input_dim):
        super(Action_Prediction, self).__init__()

        layers = [nn.Linear(input_dim, nb_hidden)]
        for _ in range(nb_layers - 1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def get_prob(self, X, batch):
        for l in self.layers:
            X = self.act(l(X))
        logits = self.final_layer(X)
        logits = logits.squeeze()

        probs = batched_softmax(logits, batch)
        return probs

    def forward(self, X, batch):
        probs = self.get_prob(X, batch)

        shifted_actions, p = batched_sample(probs, batch)
        actions = shifted_actions - get_batch_shift(batch)
        return p, actions, shifted_actions

    def evaluate(self, X, batch, actions):
        probs = self.get_prob(X, batch)

        batch_shift = get_batch_shift(batch)
        shifted_actions = actions + batch_shift
        return probs[shifted_actions]


class MyBatchNorm(torch.nn.BatchNorm1d):
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

