import torch
import torch.nn as nn

import torch_geometric as pyg

class GNN(nn.Module):
    def __init__(self, input_dim, nb_hidden, nb_layer):
        super(GNN, self).__init__()
        layers = [pyg.nn.GATConv(input_dim, nb_hidden)]
        for _ in range(nb_layer-1):
            layers.append(pyg.nn.GATConv(nb_hidden, nb_hidden))
        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        for l in self.layers:
            x = l(x, edge_index)
            x = self.act(x)
        x = pyg.nn.global_add_pool(x, g.batch)
        y = self.final_layer(x).squeeze()
        return y
