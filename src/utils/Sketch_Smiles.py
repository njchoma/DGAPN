import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import csv
import pandas as pd


def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--name', required=True)
    add_arg('--conditional', type = int, required=True)
    add_arg('--info_path', required=True)
    return parser.parse_args()


def main():
    args = read_args()

    if args.conditional:
        cols = ['start_smile', 'smile', 'tanimoto', 'reward_valid', 'reward_qed', 'reward_sa', \
                           'final_stat', 'flag_steric_strain_filter', 'flag_zinc_molecule_filter', 'stop', \
                           'surrogate_reward', 'final_reward']
    else:
        cols = ['smile', 'reward_valid', 'reward_qed', 'reward_sa', 'final_stat', 'flag_steric_strain_filter', \
                'flag_zinc_molecule_filter', 'stop', 'surrogate_reward', 'final_reward']

    data = pd.read_csv(args.info_path, names=cols, header=None)
    #print(data.head())
    filtered = data.query("flag_steric_strain_filter == False" and "stop == True")
    #print(filtered.head())
    filtered = filtered.sort_values(by='final_reward', ascending=False)
    #filtered.to_csv(str(args.name + ".csv"), index=False)


    # with open(args.info_path, 'r') as fp:
    #     reader = csv.reader(fp, delimiter=',', quotechar='"')
    #     smiles = [row[args.column] for row in reader]
    if args.conditional:
        molecules = [Chem.MolFromSmiles(i) for i in filtered['start_smile'][:10]]
        img = Draw.MolsToGridImage(molecules, subImgSize=(500, 500), molsPerRow=5, useSVG=False)
        img.save(str(args.name) + '_start.png')
        print("Tanimoto Similarity: " + str(filtered['tanimoto'][:10]))

    molecules = [Chem.MolFromSmiles(i) for i in filtered['smile'][:10]]
    img = Draw.MolsToGridImage(molecules, subImgSize=(500, 500), molsPerRow=5, useSVG=False)
    img.save(str(args.name) + '.png')

    print("Surrogate Rewards: " + str(filtered['surrogate_reward'][:10]))
    print("Final_stat: " + str(filtered['final_stat'][:10]))
    print("Final_reward: " + str(filtered['final_reward'][:10]))


if __name__ == "__main__":
    main()
