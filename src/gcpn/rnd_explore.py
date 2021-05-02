import numpy as np

import torch
import torch.nn as nn

from collections import deque

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDistillation(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 use_3d,
                 rnd_nb_layers,
                 rnd_nb_hidden,
                 rnd_nb_output):
        super(RNDistillation, self).__init__()

        self.f = RandomNetwork(emb_dim,
                               rnd_nb_output,
                               rnd_nb_layers,
                               rnd_nb_hidden,
                               init_method=init_method_1)
        self.f_hat = RandomNetwork(emb_dim,
                                   rnd_nb_output,
                                   rnd_nb_layers,
                                   rnd_nb_hidden,
                                   init_method=init_method_2)

        self.optimizer = torch.optim.Adam(self.f_hat.parameters(), lr=lr, betas=betas, eps=eps)

        self.running_error = deque(maxlen=5000)

    def forward(self, g_next_emb):
        errors = torch.norm(self.f(g_next_emb).detach() - self.f_hat(g_next_emb), dim=1)
        return errors

    def get_score(self, g_next_emb, out_min=-5., out_max=5., min_running=100, eps=0.01):
        if len(self.running_error) < min_running:
            return np.zeros(g_next_emb.size(0))

        with torch.autograd.no_grad():
            errors = self(g_next_emb).detach().cpu().numpy()
        scores = (errors - np.mean(self.running_error)) / (np.std(self.running_error) + eps)
        return np.clip(scores, out_min, out_max)

    def update(self, g_next_emb):
        errors = self(g_next_emb)
        loss = errors.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.running_error.extend(errors.detach().cpu().numpy())
        return loss.item()


class RandomNetwork(nn.Module):
    def __init__(self,
                 emb_dim,
                 output_dim,
                 nb_layers,
                 nb_hidden,
                 init_method=None):
        super(RandomNetwork, self).__init__()
        layers = []
        in_dim = emb_dim
        for _ in range(nb_layers):
            curr_layer = nn.Linear(in_dim, nb_hidden)
            if init_method is not None:
                curr_layer.apply(init_method)
            layers.append(curr_layer)
            in_dim = nb_hidden

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(in_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X)

