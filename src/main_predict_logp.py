import os
import csv
import argparse

from dataset import get_dataset
from predict_logp import predict_logp

def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')

    return parser.parse_args()

def preprocess(dataset_path):
    all_logp = []
    all_smiles = []
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, (logp, smiles) in enumerate (reader):
            all_logp.append(float(logp))
            all_smiles.append(smiles)
    return all_logp, all_smiles


def main():
    args = read_args()

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    dataset_path = get_dataset.main(args.data_path)
    logp, smiles = preprocess(dataset_path)

    predict_logp.main(artifact_path, logp, smiles)


if __name__ == "__main__":
    main()
