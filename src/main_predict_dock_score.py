import os
import argparse

from dataset import preprocess
from predict_logp import predict_logp

def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--data_path', required=True)
    add_arg('--artifact_path', default="dock_score_models")
    add_arg('--name', default="default_run")
    add_arg('--batch_size', type=int, default=256)

    return parser.parse_args()


def main():
    args = read_args()

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    logp, smiles = preprocess.main(args.data_path)

    predict_logp.main(artifact_path,
                      logp,
                      smiles,
                      'dock_score',
                      batch_size=args.batch_size)


if __name__ == "__main__":
    main()
