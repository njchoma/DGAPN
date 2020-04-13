import argparse
from dataset import get_dataset, preprocess

def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')

    return parser.parse_args()


def main():
    args = read_args()
    dataset_path = get_dataset.main(args.data_path)
    print(dataset_path)
    logp, smiles = preprocess.main(dataset_path)


if __name__ == "__main__":
    main()
