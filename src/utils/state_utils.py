import numpy as np

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from rdkit import Chem

from utils.graph_utils import *


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


def state_to_mol(state, env, keep_self_edges=True):
    nodes = state['node'].squeeze()
    nb_nodes = int(np.sum(nodes))
    adj = state['adj'][:, :nb_nodes, :nb_nodes]

    atoms = nodes_to_atom_labels(nodes, env, nb_nodes)
    bonds = []
    for a, b in zip(adj, env.possible_bond_types):
        sp = dense_to_sparse_adj(a, keep_self_edges)
        bonds.append((sp, b))

    # create empty editable mol object
    mol = Chem.RWMol()
    active_atoms = get_active_atoms(bonds)

    # add atoms to mol and keep track of index
    atom_reindex = [-1 for _ in range(len(atoms))]
    atom_count = 0
    for i, atom in enumerate(atoms):
        if i in active_atoms:
            a = Chem.Atom(atom)
            molIdx = mol.AddAtom(a)

            atom_reindex[i] = atom_count
            atom_count += 1

    for b in bonds:
        adj = b[0]
        # If molecule has no bonds of that type, continue.
        if adj.shape[1] == 0:
            continue
        adj = add_reverse(adj).transpose().tolist()
        bond_type = b[1]
        for e in adj:
            i = atom_reindex[e[0]]
            j = atom_reindex[e[1]]
            if i <= j:
                continue
            mol.AddBond(i, j, bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()

    # Ensure uniform representation based off smile string alone
    # Yes this really matters!
    #mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol is None:
        raise TypeError("Mol is None.")
    return mol


def state_to_graph(state, env, keep_self_edges=True):
    mol = state_to_mol(state, env, keep_self_edges)
    g = mol_to_pyg_graph(mol)
    g = Batch.from_data_list([g])
    return g
