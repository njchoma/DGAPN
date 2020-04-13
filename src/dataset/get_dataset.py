import os
import wget

DATASET_LOCATION="https://raw.githubusercontent.com/bowenliu16/rl_graph_generation/master/gym-molecule/gym_molecule/dataset"
DATASET_NAME="zinc_plogp_sorted.csv"

def main(data_path):
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    raw_data_filepath = os.path.join(data_path, DATASET_NAME)
    if not os.path.isfile(raw_data_filepath):
        print("Dataset not found. Downloading.")
        url = os.path.join(DATASET_LOCATION, DATASET_NAME)
        wget.download(url, raw_data_filepath)
    else:
        print("Dataset found.")
    return raw_data_filepath

