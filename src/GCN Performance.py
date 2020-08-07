#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw

import networkx as nx

from utils.graph_utils import *
from dataset.preprocess import *
from predict_logp.predict_logp import *
import torch_geometric as pyg


# # Assessing GCN Performance at the top of the ranked list

# In[2]:


import csv

def read_data(dataset_path):
    all_logp = []
    all_smiles = []
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, (logp, smiles) in enumerate (reader):
            #Some fields are empty, if logp is empty it will be caught by the exception. If smile is empty, conditional kicks in.
            try:
                if smiles is not None:
                    all_logp.append(float(logp))
                    all_smiles.append(smiles)
                else:
                    continue
            except:
                print("Row " + str(i) + "was not read.")
                continue
    return all_logp, all_smiles

def compute_baseline_error(scores):
    mean = scores.mean()
    sq_sum = np.sum(np.square(scores-mean)) / len(scores)
    return sq_sum


# In[3]:


#Loading Data
scores, smiles = read_data("/global/home/users/adchen/MD/2col/3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.2col.csv")


# In[4]:


#Np_seed remains the same so the same split is used.
train_data, valid_data, test_data = create_datasets(scores, smiles) 
test_labels = np.array(test_data.logp)

# In[5]:


batch_size = 512
num_workers = 24
train_loader = DataLoader(train_data,
                          shuffle=True,
                          collate_fn=my_collate,
                          batch_size=batch_size,
                          num_workers=num_workers)
valid_loader = DataLoader(valid_data,
                          collate_fn=my_collate,
                          batch_size=batch_size,
                          num_workers=num_workers)
test_loader =  DataLoader(test_data,
                          collate_fn=my_collate,
                          batch_size=batch_size,
                          num_workers=num_workers)


# In[6]:

#len(train_loader), len(valid_loader), len(test_loader)


# In[7]:


#len(train_data.logp), len(valid_data.logp), len(test_data.logp)


# In[8]:


#compute_baseline_error(np.array(train_data.logp)), compute_baseline_error(np.array(valid_data.logp)),compute_baseline_error(np.array(test_data.logp))


# In[39]:


if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(DEVICE)
#DEVICE = 'cpu'

# In[40]:


#Loading model, remove map_location if running on exalearn.
gcn_net = torch.load("../3CLPro_default/predict_logp/best_model.pth", map_location=DEVICE)
gcn_net.eval()

# In[ ]:
with torch.no_grad():
    pred_dock_scores = torch.empty(0)
    for i, (g1,y,g2) in enumerate(test_loader):
        g1 = g1.to(DEVICE)
        g2 = g2.to(DEVICE)
        y_pred = gcn_net(g1, g2.edge_index).cpu()
        pred_dock_scores = torch.cat((pred_dock_scores, y_pred))
        print("Batch " + str(i))
# In[40]:


#Want to plot accuracy 
#test_labels = np.array(test_scores[:1000])
sort_idx = np.argsort(test_labels)
test_labels_sorted = test_labels[sort_idx]

pred_labels = pred_dock_scores.numpy()
pred_labels_sorted = pred_labels[sort_idx]


# In[41]:


def tail_mse(a,b):
    assert len(a) == len(b)
    sq_diff = (a-b)**2
    sum_sqdiff = np.cumsum(sq_diff)
    denom = np.arange(len(a))+1
    return sum_sqdiff/denom


# In[42]:


# pred_labels = test_labels+np.random.normal(size = len(test_labels))


# In[43]:


gcn_tail_mse = tail_mse(pred_labels_sorted, test_labels_sorted)


# In[31]:


import matplotlib.pyplot as plt


# In[60]:


fig = plt.figure(figsize = (12,7))
ax = fig.add_subplot(111)
ax.plot(test_labels_sorted, gcn_tail_mse)
ax.axvline(-3.4517, color = 'red')
ax.set(title = "MSE on top fractions of Test Dataset", ylabel="MSE", xlabel="Dock Score")
plt.savefig('gcn_tail_mse.png')

