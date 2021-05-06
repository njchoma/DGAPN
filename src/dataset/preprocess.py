import csv

def main(dataset_path):
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        nb_col = len(next(reader))
        if nb_col == 2:
            scores, smiles = read_2col(reader)
        elif nb_col == 4:
            scores, smiles = read_4col(reader)
    return scores, smiles

def read_2col(reader):
    all_score = []
    all_smiles = []
    for i, (score, smiles) in enumerate (reader):
        #Some fields are empty, if logp is empty it will be caught by the exception. If smile is empty, conditional kicks in.
        try:
            if smiles is not None:
                all_score.append(float(score))
                all_smiles.append(smiles)
            else:
                continue
        except:
            print("Row " + str(i) + " was not read.")
            continue
    return all_score, all_smiles

def read_4col(reader):
    all_score = []
    all_smiles = []
    for i, (_, smiles, _, score) in enumerate (reader):
        try:
            if smiles is not None:
                all_score.append(float(score))
                all_smiles.append(smiles)
            else:
                continue
        except:
            print("Row " + str(i) + " was not read.")
            continue
    return all_score, all_smiles
