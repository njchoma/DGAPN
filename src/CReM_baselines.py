#!/usr/bin/env python
# coding: utf-8

# # CREM_Baselines
# 1. Just one round of CReM
# 2. Greedy Search: Take top 1000 molecules in training set, and continually re-apply CReM until surrogate value reaches a certain threshold.

# In[1]:


from rdkit import Chem

from crem.crem import mutate_mol

import os
import csv
import random
import numpy as np

import concurrent.futures

import torch
from torch_geometric.data import Batch

from utils.graph_utils import mol_to_pyg_graph
from utils.general_utils import load_surrogate_model, maybe_download_file


# In[2]:

# Downloading files.
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
surrogate_model_url = "https://portal.nersc.gov/project/m3623/docking_score_models/NSP15_6W01_A_1_F/20201210_170_161_baseline/best_model.pth"
training_data_url = "https://portal.nersc.gov/project/m3623/MD/2col/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.2col.csv"

maybe_download_file(os.path.join(os.getcwd(), "train_data.csv"), training_data_url, "train_data")


# In[3]:


# Loading Training Data and Surrogate Models
model = load_surrogate_model("..", surrogate_model_url, "", device)

with open(os.path.join(os.getcwd(), "train_data.csv"), 'r') as fp:
    reader = csv.reader(fp, delimiter=',', quotechar='"')
    # List of lists, [[Smile, Score, row_number],...,]
    training_data = [[row[1], row[0]] for id, row in enumerate(reader)]

training_sample = random.sample(training_data, 1000)
molecules = [Chem.MolFromSmiles(row[0]) for row in training_sample]


# In[4]:


# Defining helper functions and threshold
threshold = -15

def get_scores(states, surrogate_model, device):
    g = Batch().from_data_list([mol_to_pyg_graph(state) for state in states])
    g = g.to(device)
    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g, None)
    scores = pred_docking_score.cpu().numpy()
    return scores

def get_best(ms, model, device):
    # ms - list of molecules
    # model - surrogate_model
    pred = get_scores(ms, model, device)
    i = np.argmin(pred)
    return ms[i], pred[i]

def evolve_mol(mol):
    best_score = 0
    iter_num = 0
    while best_score > threshold and iter_num < 10:
        new_mols = list(mutate_mol(Chem.AddHs(mol), db_name='../replacements02_sc2.db', return_mol=True, min_size=0, max_size=8, min_inc=-3, max_inc=3, ncores=4))
        new_mols = [Chem.RemoveHs(i[1]) for i in new_mols]
        best_mol, score = get_best(new_mols + [mol], model, device)
        
        if score < best_score:
            best_score = score
        print('molecules generated:', len(new_mols))
        print('best score:', np.round(score, 3))
        print(best_score > threshold)
        mol = best_mol
        iter_num += 1
    print(f"Done! Score: {score}")
    return best_mol, score


# In[ ]:


# Greedy search with MultiProcessing
#with concurrent.futures.ProcessPoolExecutor() as executor:
#    results = list(executor.map(evolve_mol, molecules))


#Greedy search with for loop
results = []
for mol in molecules:
    mol, score = evolve_mol(mol)
    results.append((Chem.MolToSmiles(mol), score))

file = open('greedy_baseline.csv', 'w+', newline = '')

with file:
    write = csv.writer(file)
    write.writerows(results)
