import csv

def main(dataset_path):
    all_logp = []
    all_smiles = []
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, (logp, smiles) in enumerate (reader):
            all_logp.append(float(logp))
            all_smiles.append(smiles)
    return all_logp, all_smiles

