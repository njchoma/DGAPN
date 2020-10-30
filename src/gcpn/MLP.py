import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(Critic, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

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
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_act(self.final_layer(X)).squeeze(1)


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

