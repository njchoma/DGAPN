import os
import argparse

from dataset import get_dataset, preprocess
from sgat import train

def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--gpu', type=int, default=0)
    add_arg('--upsample', default=False)
    add_arg('--exp_loss', default=False)
    add_arg('--hidden', type=int, default=256)
    add_arg('--layers', type=int, default=3)
    add_arg('--use_3d', action='store_true')
    add_arg('--store_preprocessed', action='store_true')

    return parser.parse_args()


def main():
    args = read_args()

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    scores, smiles = preprocess.main(args.data_path)

    train.main(artifact_path,
                       scores,
                       smiles,
                       gpu_num=args.gpu,
                       upsample=args.upsample,
                       exp_loss=args.exp_loss,
                       nb_hidden=args.hidden,
                       nb_layers=args.layers,
                       use_3d=args.use_3d,
                       store_preprocessed=args.store_preprocessed,
                       data_path = args.data_path)


if __name__ == "__main__":
    main()
