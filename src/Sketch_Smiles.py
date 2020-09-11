import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import csv


def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--name', required=True)
    add_arg('--info_path', required=True)
    add_arg('--column', type=int, required=True)
    return parser.parse_args()


def main():
    args = read_args()

    with open(args.info_path, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        smiles = [row[args.column] for row in reader]

    molecules = [Chem.MolFromSmiles(i) for i in smiles]
    img = Draw.MolsToGridImage(molecules, subImgSize=(500, 500), molsPerRow=5, useSVG=False)
    img.save(str(args.name) + '.png')


if __name__ == "__main__":
    main()
