#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import logging

from utils.graph_utils import *
from dataset.preprocess import *
from predict_logp.predict_logp import *
import csv
import matplotlib.pyplot as plt


def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data_path', nargs="+", required=True)
    add_arg('--model_path', nargs="+", required=True)
    add_arg('--name', required=True)
    add_arg('--gpu', default=0)
    return parser.parse_args()


def tail_mse(a, b):
    assert len(a) == len(b)
    sq_diff = (a - b) ** 2
    sum_sqdiff = np.cumsum(sq_diff)
    denom = np.arange(len(a)) + 1
    return sum_sqdiff / denom


# Currently pearsonr calculation is too computaionally expensive.
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


def intersection_length(a, b):
    """A and B are iterables"""
    return len(set(a) & set(b))


def main():
    args = read_args()

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(args.gpu))
    else:
        DEVICE = 'cpu'
    logging.info(DEVICE)

    # Initialize logger
    logging.basicConfig(filename=str(args.name) + "_log.txt")

    data_paths = args.data_path
    models = args.model_path

    # Forward pass
    for i, (data, model_path) in enumerate(zip(data_paths, models)):

        # Loading Data
        scores, smiles = read_data(data)
        # Np_seed remains the same so the same split is used.
        train_data, valid_data, test_data = create_datasets(scores, smiles)
        test_labels = np.array(test_data.logp)
        # test_weights = torch.DoubleTensor(dock_score_weights(test_labels))
        # test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

        batch_size = 512
        num_workers = 12

        test_loader = DataLoader(test_data,
                                 collate_fn=my_collate,
                                 batch_size=batch_size,
                                 num_workers=num_workers)

        # Sorting test labels to compare performance at top of the ranked list.
        sort_idx = np.argsort(test_labels)
        test_labels_sorted = test_labels[sort_idx]

        # Storing MSE and R-squared
        gcn_tail_mses = np.empty((len(models), len(test_labels)))
        top_corrs, corrs = [], []

        # Loading model
        logging.info(model_path)
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

        # R-squared information
        top5percentidx = int(len(test_labels) // 20)
        pred_labels_top, test_labels_top = pred_labels_sorted[:top5percentidx], test_labels_sorted[:top5percentidx]
        top_corr, _ = pearsonr(pred_labels_top, test_labels_top)
        top_corrs = np.append(top_corrs, top_corr)
        corr = pearsonr(pred_labels, test_labels)[0]
        corrs = np.append(corrs, corr)

        # MSE information
        # gcn_tail_cor = tail_corr(pred_labels_sorted, test_labels_sorted)
        plot_label = models[i].split("/")[-3]
        gcn_tail_mse = tail_mse(pred_labels_sorted, test_labels_sorted)
        gcn_tail_mses[i] = gcn_tail_mse

        # Pair plots
        top5percent_shuffidx = np.random.permutation(top5percentidx)[:1000]
        pred_labels_top, test_labels_top = pred_labels_top[top5percent_shuffidx], test_labels_top[top5percent_shuffidx]

        shuff_idx = np.random.permutation(len(pred_dock_scores))[:2000]
        sample_pred_scores, sample_target_scores = pred_labels[shuff_idx], test_labels[shuff_idx]

        # Overlap of top 10 and top 100

        fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
        plt.suptitle(str(plot_label))
        ax[0].scatter(pred_labels_top, test_labels_top, c="Blue", label="Top 5%")
        ax[0].set_xlabel("Predicted Scores")
        ax[0].set_ylabel("Target Scores")
        ax[1].scatter(sample_pred_scores, sample_target_scores, c="Blue", label="Whole Dataset")
        ax[1].set_xlabel("Predicted Scores")
        ax[1].set_ylabel("Target Scores")
        fig.savefig(str(plot_label) + '_pairplots.png')
        fig.clf()

    # Saving Diagnostics to log.txt
    logging.info("Models " + str([model.split("/")[-3] for model in models]))
    logging.info("Top 5% r-squared: " + str(top_corrs))
    logging.info("Overall r-squared: " + str(corrs))
    logging.info("Top 100 Intersection " + str(intersection_length(pred_labels_sorted[:100], test_labels_sorted[:100])))
    logging.info("Top 10 Intersection " + str(intersection_length(pred_labels_sorted[:10], test_labels_sorted[:10])))

    # After looping through all models and calculating tail_mse, plot on one plot.
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for i, model_path in enumerate(models):
        plot_label = model_path.split("/")[-3]
        ax.plot(test_labels_sorted, gcn_tail_mses[i], label=plot_label)

    # ax.plot(test_labels_sorted, gcn_tail_cor, c="Orange", label="Cor")
    # ax.axvline(-3.4517, color='red')
    ax.axhline(compute_baseline_error(test_labels), color='purple')
    ax.legend()
    ax.set(title="MSE on top fractions of Test Dataset", ylabel="MSE", xlabel="Dock Score")
    plt.savefig(str(args.name) + '_tail_mse.png')


if __name__ == "__main__":
    main()
