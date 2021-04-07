import numpy as np

import torch
import torch.nn as nn

from collections import deque

EPS = 1e-2

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDistillation(nn.Module):
    def __init__(self, lr, betas, eps, emb_dim, output_dim, nb_layers, nb_hidden):
        super(RNDModel, self).__init__()

        self.f = RandomNetwork(emb_dim, output_dim, nb_layers, nb_hidden, init_method=init_method_1)
        self.f_hat = RandomNetwork(emb_dim, output_dim, nb_layers, nb_hidden, init_method=init_method_2)

        self.optimizer = torch.optim.Adam(self.f_hat.parameters(), lr=lr, betas=betas, eps=eps)

        self.running_error = deque(maxlen=5000)

    def forward(self, X):
        errors = torch.norm(self.f(X).detach() - self.f_hat(X), dim=1).squeeze(1)
        return errors

    def get_score(self, X):
        with torch.autograd.no_grad():
            errors = self(X).detach().cpu().numpy()
        scores = (errors - np.mean(self.running_error)) / (np.std(self.running_error) + EPS)
        return scores

    def update(self, X):
        loss = self(X).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class RandomNetwork(nn.Module):
    def __init__(self, emb_dim, output_dim, nb_layers, nb_hidden, init_method=None):
        super(RandomNetwork, self).__init__()
        layers = []
        in_dim = emb_dim
        for _ in range(n_layers):
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

