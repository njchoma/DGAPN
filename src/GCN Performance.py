#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.graph_utils import *
from dataset.preprocess import *
from predict_logp.predict_logp import *
import csv


# # Assessing GCN Performance at the top of the ranked list
def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data_path', required=True)

    return parser.parse_args()

def read_data(dataset_path):
    all_logp = []
    all_smiles = []
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for i, (logp, smiles) in enumerate(reader):
            # Some fields are empty, if logp is empty it will be caught by the exception. If smile is empty,
            # conditional kicks in.
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
    sq_sum = np.sum(np.square(scores - mean)) / len(scores)
    return sq_sum


# Loading Data
scores, smiles = read_data("/global/home/users/adchen/MD/2col/3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.2col.csv")

# Np_seed remains the same so the same split is used.
train_data, valid_data, test_data = create_datasets(scores, smiles)
test_labels = np.array(test_data.logp)

test_weights = torch.DoubleTensor(dock_score_weights(test_labels))
test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

batch_size = 512
num_workers = 24
# train_loader = DataLoader(train_data,
#                           shuffle=True,
#                           collate_fn=my_collate,
#                           batch_size=batch_size,
#                           num_workers=num_workers)
# valid_loader = DataLoader(valid_data,
#                           collate_fn=my_collate,
#                           batch_size=batch_size,
#                           num_workers=num_workers)
test_loader = DataLoader(test_data,
                         collate_fn=my_collate,
                         batch_size=batch_size,
                         num_workers=num_workers)

# print(compute_baseline_error(np.array(train_data.logp)), compute_baseline_error(np.array(valid_data.logp)),\
#       compute_baseline_error(np.array(test_data.logp)))


if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(DEVICE)

# DEVICE = 'cpu'

# Loading model, remove map_location if running on exalearn.
gcn_net = torch.load("../3CLPro_default/predict_logp/best_model.pth", map_location=DEVICE)
gcn_net.eval()

with torch.no_grad():
    pred_dock_scores = torch.empty(0)
    for i, (g1, y, g2) in enumerate(test_loader):
        g1 = g1.to(DEVICE)
        g2 = g2.to(DEVICE)
        y_pred = gcn_net(g1, g2.edge_index).cpu()
        pred_dock_scores = torch.cat((pred_dock_scores, y_pred))
        print("Batch " + str(i))

# Want to plot accuracy


# Want to plot accuracy
# test_labels = np.array(test_scores[:1000])
sort_idx = np.argsort(test_labels)
test_labels_sorted = test_labels[sort_idx]

pred_labels = pred_dock_scores.numpy()
pred_labels_sorted = pred_labels[sort_idx]

from scipy.stats import pearsonr


def tail_mse(a, b):
    assert len(a) == len(b)
    sq_diff = (a - b) ** 2
    sum_sqdiff = np.cumsum(sq_diff)
    denom = np.arange(len(a)) + 1
    return sum_sqdiff / denom


def tail_corr(a, b):
    assert len(a) == len(b)
    tail_corr = np.array([pearsonr(a[:i], b[:i])[0] for i in np.arange(30, len(a))], dtype=float)
    tail_corr = np.insert(tail_corr, 0, np.repeat(np.nan, 30))
    return tail_corr


pred_labels_11, test_labels_11 = pred_labels_sorted[test_labels_sorted < -11], test_labels_sorted[
    test_labels_sorted < -11]
corr, _ = pearsonr(pred_labels_11, test_labels_11)
print("R-squared: " + str(corr ** 2))

gcn_tail_cor = tail_corr(pred_labels_sorted, test_labels_sorted)
gcn_tail_mse = tail_mse(pred_labels_sorted, test_labels_sorted)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111)
ax.plot(test_labels_sorted, gcn_tail_mse, c="Blue", label="MSE")
ax.plot(test_labels_sorted, gcn_tail_cor, c="Orange", label="Cor")
ax.axvline(-3.4517, color='red')
ax.axhline(1.0405, color='purple')
ax.legend()
ax.set(title="MSE on top fractions of Test Dataset", ylabel="MSE", xlabel="Dock Score")
plt.savefig('gcn_tail_mse.png')
