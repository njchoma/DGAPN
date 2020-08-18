#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem import Draw

from utils.graph_utils import *
from predict_logp.predict_logp import *

def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--model_path', nargs="+", required=True)
    add_arg('--name', required=True)
    add_arg('--gpu', default=0)
    return parser.parse_args()


def main():
    args = read_args()

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(args.gpu))
    else:
        DEVICE = 'cpu'
    print(DEVICE)

    # # Loading generated molecules
    colnames = ['SMILE', 'rew_valid', 'rew_qed', 'rew_sa', 'final_stat', 'rew_env', 'rew_d_step', 'rew_d_final', \
                'cur_ep_et', 'flag_steric_strain_filter', 'flag_zinc_molecule_filter', 'stop']
    df_logpen = pd.read_csv('/global/home/users/adchen/rl_graph_generation/molecule_gen/molecule_zinc_logppen.csv',
                            header=None, names=colnames)
    df_qed_condition = pd.read_csv(
        '/global/home/users/adchen/rl_graph_generation/molecule_gen/molecule_zinc_qed_conditional.csv', header=None,
        names=colnames)
    df_qedsa = pd.read_csv('/global/home/users/adchen/rl_graph_generation/molecule_gen/molecule_zinc_qedsa.csv',
                           header=None, names=colnames)
    df_qed = pd.read_csv('/global/home/users/adchen/rl_graph_generation/molecule_gen/molecule_zinc_test_conditional.csv',
                         header=None, names=colnames)

    df_logpen = df_logpen[~df_logpen["SMILE"].str.contains("Iteration")]
    df_qed_condition = df_qed_condition[~df_qed_condition["SMILE"].str.contains("Iteration")]
    df_qedsa = df_qedsa[~df_qedsa["SMILE"].str.contains("Iteration")]
    df_qed = df_qed[~df_qed["SMILE"].str.contains("Iteration")]

    # Filter by steric_strain_filter == True, flag_zinc_molecule_filter==True, and sort by final reward
    mol_filter = 'flag_steric_strain_filter == True & flag_zinc_molecule_filter == True'
    df_logpen = df_logpen.query(mol_filter).sort_values("final_stat", ascending=False)
    df_qed_condition = df_qed_condition.query(mol_filter).sort_values("final_stat", ascending=False)
    df_qedsa = df_qedsa.query(mol_filter).sort_values("final_stat", ascending=False)
    df_qed = df_qed.query(mol_filter).sort_values("final_stat", ascending=False)

    logpen_smiles = df_logpen["SMILE"].values[:10000]
    qed_condition_smiles = df_qed_condition["SMILE"].values[:10000]
    qedsa_smiles = df_qedsa["SMILE"].values[:10000]
    qed_smiles = df_qed["SMILE"].values[:10000]

    # # Forward pass on GCN
    gcn_net = torch.load(args.model_path, map_location=DEVICE)

    logpen_data = MolData([0] * len(logpen_smiles), logpen_smiles)
    qed_condition_data = MolData([0] * len(qed_condition_smiles), qed_condition_smiles)
    qedsa_data = MolData([0] * len(qedsa_smiles), qedsa_smiles)
    qed_data = MolData([0] * len(qed_smiles), qed_smiles)


    logpen_dataloader = DataLoader(logpen_data, collate_fn=my_collate, batch_size=512, num_workers=24)
    qed_condition_dataloader = DataLoader(qed_condition_data, collate_fn=my_collate, batch_size=512, num_workers=24)
    qedsa_dataloader = DataLoader(qedsa_data, collate_fn=my_collate, batch_size=512, num_workers=24)
    qed_dataloader = DataLoader(qed_data, collate_fn=my_collate, batch_size=512, num_workers=24)

    with torch.no_grad():
        logpen_scores = torch.empty(0)
        qed_condition_scores = torch.empty(0)
        qedsa_scores = torch.empty(0)
        qed_scores = torch.empty(0)

        for i, (g1, y, g2) in enumerate(logpen_dataloader):
            g1 = g1.to(DEVICE)
            g2 = g2.to(DEVICE)
            y_pred = gcn_net(g1, g2.edge_index).cpu()
            logpen_scores = torch.cat((logpen_scores, y_pred))
            print("Batch " + str(i))

        for i, (g1, y, g2) in enumerate(qed_condition_dataloader):
            g1 = g1.to(DEVICE)
            g2 = g2.to(DEVICE)
            y_pred = gcn_net(g1, g2.edge_index).cpu()
            qed_condition_scores = torch.cat((qed_condition_scores, y_pred))
            print("Batch " + str(i))

        for i, (g1, y, g2) in enumerate(qedsa_dataloader):
            g1 = g1.to(DEVICE)
            g2 = g2.to(DEVICE)
            y_pred = gcn_net(g1, g2.edge_index).cpu()
            qedsa_scores = torch.cat((qedsa_scores, y_pred))
            print("Batch " + str(i))

        for i, (g1, y, g2) in enumerate(qed_dataloader):
            g1 = g1.to(DEVICE)
            g2 = g2.to(DEVICE)
            y_pred = gcn_net(g1, g2.edge_index).cpu()
            qed_scores = torch.cat((qed_scores, y_pred))
            print("Batch " + str(i))

    logpen_scores = logpen_scores.numpy()
    qed_condition_scores = qed_condition_scores..numpy()
    qedsa_scores = qedsa_scores.numpy()
    qed_scores = qed_scores.numpy()

    top_logpen_mols = logpen_scores.argsort()[:6]
    top_qed_condition_mols = qed_condition_scores.argsort()[:6]
    top_qedsa_mols = qedsa_scores.argsort()[:6]
    top_qed_mols = qed_scores.argsort()[:6]


    top_logpen_smiles = logpen_smiles[top_logpen_mols]
    top_qed_condition_smiles = qed_condition_smiles[top_qed_condition_mols]
    top_qedsa_smiles = qedsa_smiles[top_qedsa_mols]
    top_qed_smiles = qed_smiles[top_qed_mols]


    top_logpen_molecules = [Chem.MolFromSmiles(i) for i in top_logpen_smiles]
    top_qed_condition_molecules = [Chem.MolFromSmiles(i) for i in top_qed_condition_smiles]
    top_qedsa_molecules = [Chem.MolFromSmiles(i) for i in top_qedsa_smiles]
    top_qed_molecules = [Chem.MolFromSmiles(i) for i in top_qed_smiles]

    img = Draw.MolsToGridImage(top_logpen_molecules, subImgSize=(300, 300), molsPerRow=3, useSVG=False)
    img.save(str(args.name) + ' ToplogpenMolecules.png')
    img = Draw.MolsToGridImage(top_qed_condition_molecules, subImgSize=(300, 300), molsPerRow=3, useSVG=False)
    img.save(str(args.name)+' TopqedcondMolecules.png')
    img = Draw.MolsToGridImage(top_qedsa_molecules, subImgSize=(300, 300), molsPerRow=3, useSVG=False)
    img.save(str(args.name)+' TopqedsaMolecules.png')
    img = Draw.MolsToGridImage(top_qed_molecules, subImgSize=(300, 300), molsPerRow=3, useSVG=False)
    img.save(str(args.name)+' TopqedMolecules.png')

if __name__ == "__main__":
    main()