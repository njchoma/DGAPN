#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
import os
import pickle

from chemprop.train import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers

rdBase.DisableLog('rdApp.error')

REW_SCALE_FACTOR = 15.0

class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'src/reward/gsk3_jnk3/data/gsk3/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, mol_list):
        fps = []
        mask = []
        for i,mol in enumerate(mol_list):
            mask.append( int(mol is not None) )
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores).tolist()

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
    

class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = 'src/reward/gsk3_jnk3/data/jnk3/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, mol_list):
        fps = []
        mask = []
        for i,mol in enumerate(mol_list):
            mask.append( int(mol is not None) )
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores).tolist()

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


def score_mols(states, scorer):
    if not isinstance(states, list):
        states = [states]
    return [s * REW_SCALE_FACTOR for s in scorer(states)]


def get_jnk3(states):
    scorer = jnk3_model()
    return score_mols(states, scorer)

def get_gsk3(states):
    scorer = gsk3_model()
    return score_mols(states, scorer)

def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""
    if prop_name == 'jnk3':
        return jnk3_model()
    elif prop_name == 'gsk3':
        return gsk3_model()
    elif prop_name == 'qed':
        return qed_func()
    elif prop_name == 'sa':
        return sa_func()
    else:
        return chemprop_model(prop_name)


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    mols = [Chem.MolFromSmiles(m) for m in all_y]
    props = [func(mols) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)
