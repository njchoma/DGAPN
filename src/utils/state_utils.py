import numpy as np

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from utils.graph_utils import state_to_pyg

def wrap_state(ob):
    # deprecated
    adj = ob['adj']
    nodes = ob['node'].squeeze()

    adj = torch.Tensor(adj)
    nodes = torch.Tensor(nodes)

    adj = [dense_to_sparse(a) for a in adj]
    data = Data(x=nodes, edge_index=adj[0][0], edge_attr=adj[0][1])
    return data

def nodes_to_atom_labels(nodes, env, nb_nodes):
    atom_types = env.possible_atom_types
    atom_idx = np.argmax(nodes[:nb_nodes], axis=1)
    node_labels = np.asarray(atom_types)[atom_idx]
    return node_labels

def dense_to_sparse_adj(adj, keep_self_edges):
    # Remove self-edges converting to surrogate input
    if not keep_self_edges:
        adj = adj - np.diag(np.diag(adj))
    sp = np.nonzero(adj)
    sp = np.stack(sp)
    return sp

def state_to_graph(state, env, keep_self_edges=True):
    nodes = state['node'].squeeze()
    nb_nodes = int(np.sum(nodes))
    adj = state['adj'][:,:nb_nodes, :nb_nodes]

    atoms = nodes_to_atom_labels(nodes, env, nb_nodes)
    bonds = []
    for a,b in zip(adj, env.possible_bond_types):
        sp = dense_to_sparse_adj(a, keep_self_edges)
        bonds.append((sp, b))
    graph = state_to_pyg(atoms, bonds)
    graph = [Batch.from_data_list([g]) for g in graph]
    return graph
