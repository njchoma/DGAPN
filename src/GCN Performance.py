#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

from utils.graph_utils import *
from dataset.preprocess import *
from predict_logp.predict_logp import *
import csv
import matplotlib.pyplot as plt


# # Assessing GCN Performance at the top of the ranked list
def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data_path', required=True)
    add_arg('--model_path', nargs="+", required=True)
    add_arg('--gpu', default=0)
    return parser.parse_args()


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


def main():
    args = read_args()
    # Loading Data
    scores, smiles = read_data(args.data_path)

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
                             sampler=test_sampler,
                             num_workers=num_workers)

    # print(compute_baseline_error(np.array(train_data.logp)), compute_baseline_error(np.array(valid_data.logp)),\
    #       compute_baseline_error(np.array(test_data.logp)))

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(args.gpu))
    else:
        DEVICE = 'cpu'
    print(DEVICE)

    sort_idx = np.argsort(test_labels)
    test_labels_sorted = test_labels[sort_idx]

    # Loading model
    models = args.model_path
    gcn_tail_mses = np.empty((len(models), len(test_labels)))

    for i, model_path in enumerate(models):
        print(model_path)
        gcn_net = torch.load(model_path, map_location=DEVICE)
        gcn_net.eval()

        with torch.no_grad():
            pred_dock_scores = torch.empty(0)
            for j, (g1, y, g2) in enumerate(test_loader):
                g1 = g1.to(DEVICE)
                g2 = g2.to(DEVICE)
                y_pred = gcn_net(g1, g2.edge_index).cpu()
                pred_dock_scores = torch.cat((pred_dock_scores, y_pred))
                print("Batch " + str(j))

        # Want to plot accuracy
        # test_labels = np.array(test_scores[:1000])

        pred_labels = pred_dock_scores.numpy()
        pred_labels_sorted = pred_labels[sort_idx]

        top5percentidx = int(len(test_labels)//20)
        pred_labels_top, test_labels_top = pred_labels_sorted[:top5percentidx], test_labels_sorted[:top5percentidx]
        corr, _ = pearsonr(pred_labels_top, test_labels_top)
        print("R-squared: " + str(corr ** 2))

        #gcn_tail_cor = tail_corr(pred_labels_sorted, test_labels_sorted)
        gcn_tail_mse = tail_mse(pred_labels_sorted, test_labels_sorted)
        gcn_tail_mses[i] = gcn_tail_mse

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.plot(test_labels_sorted, gcn_tail_mses[0], c="Blue", label="MSE_default")
    ax.plot(test_labels_sorted, gcn_tail_mses[1], c="Red", label="MSE_upsampled")
    ax.plot(test_labels_sorted, gcn_tail_mses[2], c="Olive", label="MSE_exp")
    #ax.plot(test_labels_sorted, gcn_tail_cor, c="Orange", label="Cor")
    #ax.axvline(-3.4517, color='red')
    ax.axhline(compute_baseline_error(test_labels), color='purple')
    ax.legend()
    ax.set(title="MSE on top fractions of Test Dataset", ylabel="MSE", xlabel="Dock Score")
    plt.savefig('gcn_tail_mse.png')


if __name__ == "__main__":
    main()