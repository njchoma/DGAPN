import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data

HIGHEST_ATOMIC_NUMBER = 118


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def atom_to_node(atom):
    idx = atom.GetIdx()
    symbol = atom.GetSymbol()
    atom_nb = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    implicit_valence = atom.GetImplicitValence()
    ring_atom = atom.IsInRing()
    degree = atom.GetDegree()
    hybridization = atom.GetHybridization()
    # print(idx, symbol, atom_nb)
    # print(ring_atom)
    # print(degree)
    # print(hybridization)
    node = [idx, atom_nb, formal_charge, implicit_valence, ring_atom]
    return node


def bond_to_edge(bond):
    src = bond.GetBeginAtomIdx()
    dst = bond.GetEndAtomIdx()
    bond_type = bond.GetBondTypeAsDouble()
    edge = [src, dst, bond_type]
    return edge


def is_sorted(l):
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1))


def construct_graph(nodes, edges):
    # NODES
    atom_num = torch.LongTensor(nodes[:, 0])
    atom_num_oh = torch.nn.functional.one_hot(atom_num, HIGHEST_ATOMIC_NUMBER)
    node_feats = torch.FloatTensor(nodes[:, 1:])
    x = torch.cat((node_feats, atom_num_oh.to(torch.float)), dim=1)

    # EDGES
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.LongTensor([src, dst])

    edge_attr = torch.FloatTensor([e[2] for e in edges])

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return g


def mol_to_pyg_graph(mol):
    nodes = []
    for atom in mol.GetAtoms():
        nodes.append(atom_to_node(atom))
    idx = [n[0] for n in nodes]
    assert is_sorted(idx)
    nodes = np.array(nodes, dtype=float)[:, 1:]
    edges = []
    for bond in mol.GetBonds():
        edges.append(bond_to_edge(bond))

    g = construct_graph(nodes, edges)
    return g


def add_reverse(orig_adj):
    adj = orig_adj.transpose()
    adj2 = np.array([adj[:, 1], adj[:, 0]]).transpose()
    all_adj = np.concatenate((adj, adj2), axis=0)
    all_adj = np.unique(all_adj, axis=0).transpose()
    return all_adj


def get_active_atoms(bonds):
    '''
    Atom 0 is always active. Only include other atoms if there is a bond connected to them.
    '''
    active_atoms = np.array([0])
    for (b, bond_type) in bonds:
        active_atoms = np.append(active_atoms, b.flatten())
    return np.unique(active_atoms)
