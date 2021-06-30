from collections import deque
import numpy as np

import torch
import torch.nn as nn

import torch_geometric as pyg

from gnn_embed import sGAT

def init_network(model, method='uniform'):
    if model is not None:
        if method == 'uniform':
            model.weight.data.uniform_()
            model.bias.data.uniform_()
        elif method == 'normal':
            model.weight.data.normal_()
            model.bias.data.normal_()
        else:
            pass


class RNDistillation(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 rnd_nb_layers,
                 rnd_nb_hidden,
                 rnd_nb_output):
        super(RNDistillation, self).__init__()

        self.f = RandomNetwork(input_dim,
                               nb_edge_types,
                               use_3d,
                               gnn_nb_layers,
                               gnn_nb_hidden,
                               rnd_nb_layers,
                               rnd_nb_hidden,
                               rnd_nb_output,
                               init_method='uniform')
        self.f_hat = RandomNetwork(input_dim,
                                   nb_edge_types,
                                   use_3d,
                                   gnn_nb_layers,
                                   gnn_nb_hidden,
                                   rnd_nb_layers,
                                   rnd_nb_hidden,
                                   rnd_nb_output,
                                   init_method='normal')

        self.optimizer = torch.optim.Adam(self.f_hat.parameters(), lr=lr, betas=betas, eps=eps)

        self.running_error = deque(maxlen=5000)

    def forward(self, states_next):
        errors = torch.norm(self.f(states_next).detach() - self.f_hat(states_next), dim=1)
        return errors

    def get_score(self, states_next, out_min=-5., out_max=5., min_running=100, eps=0.01):
        with torch.autograd.no_grad():
            errors = self(states_next).detach().cpu().numpy()

        if len(self.running_error) < min_running:
            return np.zeros_like(errors)
        scores = (errors - np.mean(self.running_error)) / (np.std(self.running_error) + eps)
        return np.clip(scores, out_min, out_max)

    def update(self, states_next):
        errors = self(states_next)
        loss = errors.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.running_error.extend(errors.detach().cpu().numpy())
        return loss.item()


class RandomNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 nb_edge_types,
                 use_3d,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 rnd_nb_layers,
                 rnd_nb_hidden,
                 rnd_nb_output,
                 init_method=None):
        super(RandomNetwork, self).__init__()
        if not isinstance(gnn_nb_hidden, list):
            gnn_nb_hidden = [gnn_nb_hidden] * gnn_nb_layers
        if not isinstance(rnd_nb_hidden, list):
            rnd_nb_hidden = [rnd_nb_hidden] * rnd_nb_layers
        else:
            assert len(rnd_nb_hidden) == rnd_nb_layers

        # gnn encoder
        self.gnn = sGAT(input_dim, nb_edge_types, gnn_nb_hidden, gnn_nb_layers, 
                        use_3d=use_3d, init_method=init_method)
        if gnn_nb_layers == 0:
            in_dim = input_dim
        else:
            in_dim = gnn_nb_hidden[-1]

        # mlp encoder
        layers = []
        for i in range(rnd_nb_layers):
            curr_layer = nn.Linear(in_dim, rnd_nb_hidden[i])
            if init_method is not None:
                init_network(curr_layer, init_method)
            layers.append(curr_layer)
            in_dim = rnd_nb_hidden[i]

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(in_dim, rnd_nb_output)
        self.act = nn.ReLU()

    def forward(self, states_next):
        X = self.gnn.get_embedding(states_next, detach=False)
        for l in self.layers:
            X = self.act(l(X))
        return self.final_layer(X)
