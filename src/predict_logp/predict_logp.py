import os
import logging
import numpy as np

from rdkit import Chem

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric as pyg


import utils.graph_utils as graph_utils
import utils.general_utils as general_utils
from .model import GNN

#############################################
#                   DATA                    #
#############################################

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

    def get_input_dim(self):
        g, y = self[0]
        input_dim = g.x.shape[1]
        return input_dim

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


#################################################
#                   TRAINING                    #
#################################################



#############################################
#                   MAIN                    #
#############################################

def main(artifact_path, logp, smiles):
    artifact_path = os.path.join(artifact_path, 'predict_logp')
    os.makedirs(artifact_path, exist_ok=True)
    general_utils.initialize_logger(artifact_path)

    train_data, valid_data, test_data = create_datasets(logp, smiles)

    net = GNN(train_data.get_input_dim())
    logging.info(net)

    general_utils.close_logger()
