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
    # # Forward pass on GCN
    gcn_net = torch.load(args.model_path, map_location=DEVICE)

    docking_data = MolData(scores, smiles)
    docking_dataloader = DataLoader(docking_data, collate_fn=my_collate, batch_size=512, num_workers=24)

    with torch.no_grad():
        pred_scores = torch.empty(0)
        for i, (g1, y, g2) in enumerate(docking_dataloader):
            g1 = g1.to(DEVICE)
            g2 = g2.to(DEVICE)
            y_pred = gcn_net(g1, g2.edge_index).cpu()
            pred_scores = torch.cat((pred_scores, y_pred))
            print("Batch " + str(i))

    pred_scores = pred_scores.numpy()

    top_gcn_scores = pred_scores.argsort()[:6]
    top_gcn_smiles = np.array(smiles)[top_gcn_scores]
    top_gcn_molecules = [Chem.MolFromSmiles(i) for i in top_gcn_smiles]

    img = Draw.MolsToGridImage(top_gcn_molecules, subImgSize=(300, 300), molsPerRow=3, useSVG=False)
    img.save(str(args.name) + ' GCN_TopZincMolecules.png')


if __name__ == "__main__":
    main()
