import os
import numpy as np
from rdkit import Chem
from crem.crem import mutate_mol

from torch_geometric.data import Batch

from dataset.get_dataset import download_dataset, unpack_dataset
from dataset import preprocess
from utils.graph_utils import mol_to_pyg_graph

DATASET_URL = "http://www.qsar4u.com/files/cremdb/replacements02_sc2.db.gz"
DATASET_NAME = 'replacements02_sc2.db'

class CReM_Env(object):
    def __init__(self,
                 storage_path,
                 nb_sample_crem = 16,
                 nb_cores = 16):

        warm_start_dataset_path = os.path.join(storage_path, 'NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv')
        self.scores, self.smiles = preprocess.main(warm_start_dataset_path)
        self.nb_sample_crem = nb_sample_crem
        self.nb_cores = nb_cores

        _ = download_dataset(storage_path,
                             DATASET_NAME+'.gz',
                             DATASET_URL)
        self.db_fname = unpack_dataset(storage_path,
                                       DATASET_NAME+'.gz',
                                       DATASET_NAME)


    def reset(self, include_current_state=True):
        idx = np.random.randint(len(self.scores))
        mol = Chem.MolFromSmiles(self.smiles[idx])
        return self.mol_to_candidates(mol, include_current_state)

    def step(self, action, include_current_state=True):
        mol = self.new_mols[action]
        return self.mol_to_candidates(mol, include_current_state)


    def mol_to_candidates(self, mol, include_current_state):
        g = mol_to_pyg_graph(mol)[0]
        g_candidates, done = self.get_crem_candidates(mol, include_current_state)

        return g, g_candidates, done

    def get_crem_candidates(self, mol, include_current_state):

        try:
            new_mols = list(mutate_mol(mol,
                                       self.db_fname,
                                       max_replacements = self.nb_sample_crem,
                                       return_mol=True,
                                       ncores=self.nb_cores))
            # print("CReM options:" + str(len(new_mols)))
            new_mols = [Chem.RemoveHs(i[1]) for i in new_mols]
        except Exception as e:
            print("CReM forward error: " + str(e))
            print("SMILE: " + Chem.MolToSmiles(mol))
            new_mols = []
        self.new_mols = [mol] + new_mols if include_current_state else new_mols
        g_candidates = [mol_to_pyg_graph(i)[0] for i in self.new_mols]
        
        if len(g_candidates)==0:
            return None, True

        g_candidates = Batch.from_data_list(g_candidates)
        return g_candidates, False
