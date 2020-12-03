import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import GraphDescriptors

import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

HIGHEST_ATOMIC_NUMBER=118

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
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

def construct_graph(nodes, edges):
    # NODES
    atom_num = torch.LongTensor(nodes[:,0])
    atom_num_oh = torch.nn.functional.one_hot(atom_num, HIGHEST_ATOMIC_NUMBER)
    node_feats = torch.FloatTensor(nodes[:,1:])
    x = torch.cat((node_feats, atom_num_oh.to(torch.float)), dim=1)

    # EDGES
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.LongTensor([src, dst])

    edge_attr = torch.FloatTensor([e[2] for e in edges])

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return g

def mol_to_pyg_graph(mol, idm=False):
    nodes = []
    for atom in mol.GetAtoms():
        nodes.append(atom_to_node(atom))
    idx = [n[0] for n in nodes]
    assert is_sorted(idx)
    nodes = np.array(nodes, dtype=float)[:,1:]
    edges = []
    for bond in mol.GetBonds():
        edges.append(bond_to_edge(bond))
    
    g_adj = construct_graph(nodes, edges)


    if idm:
        # inverse distance weighting matrix
        # mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == -1:  # optional random seed for reproducibility)
            AllChem.Compute2DCoords(mol)
        # mol = Chem.RemoveHs(mol)
        W = 1. / Chem.rdmolops.Get3DDistanceMatrix(mol)
        W[np.isinf(W)] = 0
        W_spr = dense_to_sparse(torch.FloatTensor(W))
        g_idm = Data(x=g_adj.x, edge_index=W_spr[0], edge_attr=W_spr[1])

        return [g_adj, g_idm] 
    return [g_adj]

def add_reverse(orig_adj):
    adj = orig_adj.transpose()
    adj2 = np.array([adj[:,1], adj[:,0]]).transpose()
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

def state_to_pyg(atoms, bonds):

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
            if i<=j:
                continue
            mol.AddBond(i, j, bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    # Ensure uniform representation based off smile string alone
    # Yes this really matters!
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol is None:
        raise TypeError("Mol is None.")
    return mol_to_pyg_graph(mol)
