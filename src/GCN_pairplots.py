#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import torch
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from utils.graph_utils import *
from predict_logp.predict_logp import *
import matplotlib.pyplot as plt


def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data_path', required=True)
    add_arg('--model_path', required=True)
    add_arg('--name', required=True)
    add_arg('--gpu', default=0)
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


def main():
    args = read_args()

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(args.gpu))
    else:
        DEVICE = 'cpu'
    print(DEVICE)

    scores, smiles = read_data(args.data_path)
    train_data, valid_data, test_data = create_datasets(scores, smiles)
    test_labels = np.array(test_data.logp)
    batch_size = 512
    num_workers = 24
    test_loader = DataLoader(test_data,
                             collate_fn=my_collate,
                             batch_size=batch_size,
                             num_workers=num_workers)
    # # Forward pass on GCN
    gcn_net = torch.load(args.model_path, map_location=DEVICE)

    with torch.no_grad():
        pred_dock_scores = torch.empty(0)
        for j, (g1, y, g2) in enumerate(test_loader):
            g1 = g1.to(DEVICE)
            g2 = g2.to(DEVICE)
            y_pred = gcn_net(g1, g2.edge_index).cpu()
            pred_dock_scores = torch.cat((pred_dock_scores, y_pred))
            print("Batch " + str(j))

    sort_idx = np.argsort(test_labels)
    test_labels_sorted = test_labels[sort_idx]
    pred_labels = pred_dock_scores.numpy()

    pred_labels_sorted = pred_labels[sort_idx]
    top5percentidx = int(len(test_labels) // 20)
    top5percent_shuffidx = np.random.permutation(top5percentidx)[:1000]
    pred_labels_top, test_labels_top = pred_labels_sorted[:top5percentidx][top5percent_shuffidx], \
                                       test_labels_sorted[:top5percentidx][top5percent_shuffidx]

    shuff_idx = np.random.permutation(len(pred_dock_scores))[:2000]
    sample_pred_scores, sample_target_scores = pred_labels[shuff_idx], test_labels[shuff_idx]

    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    ax[0].scatter(pred_labels_top, test_labels_top, c="Blue", label="Top 5%")
    ax[0].set_xlabel("Predicted Scores")
    ax[0].set_ylabel("Target Scores")
    ax[1].scatter(sample_pred_scores, sample_target_scores, c="Blue", label="Whole Dataset")
    ax[1].set_xlabel("Predicted Scores")
    ax[1].set_ylabel("Target Scores")
    fig.savefig(str(args.name) + '_pairplots.png')


if __name__ == "__main__":
    main()
