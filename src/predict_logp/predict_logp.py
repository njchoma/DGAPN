import numpy as np

from rdkit import Chem

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric as pyg


import utils.graph_utils as graph_utils

class MolData(Dataset):
    def __init__(self, logp, smiles):
        super(MolData, self).__init__()
        self.logp = logp
        self.smiles = smiles

    def __getitem__(self, index):
        logp = self.logp[index]
        smiles = self.smiles[index]

        mol = Chem.MolFromSmiles(smiles)
        g = graph_utils.mol_to_pyg_graph(mol)
        return g, torch.FloatTensor([logp])

    def __len__(self):
        return len(self.logp)

def create_datasets(logp, smiles):
    nb_samples = len(logp)
    assert nb_samples > 10

    nb_train = int(nb_samples * 0.6)
    nb_valid = int(nb_samples * 0.2)

    sample_order = np.random.permutation(nb_samples)

    logp = np.asarray(logp)[sample_order].tolist()
    smiles = np.asarray(smiles)[sample_order].tolist()

    train_data = MolData(logp[:nb_train], smiles[:nb_train])
    valid_data = MolData(logp[nb_train:nb_train+nb_valid],
                         smiles[nb_train:nb_train+nb_valid])
    test_data  = MolData(logp[nb_train+nb_valid:], smiles[nb_train+nb_valid:])
    return train_data, valid_data, test_data


def main(logp, smiles):
    train_data, valid_data, test_data = create_datasets(logp, smiles)
    g, y = train_data[1]
    print(g, y)
