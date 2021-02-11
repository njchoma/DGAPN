import os
import wget
import gzip
import shutil

DATASET_LOCATION="https://raw.githubusercontent.com/bowenliu16/rl_graph_generation/master/gym-molecule/gym_molecule/dataset"
DATASET_NAME="zinc_plogp_sorted.csv"

def download_dataset(storage_path, dataset_filename, dataset_url):
    if not os.path.isdir(storage_path):
        os.mkdir(storage_path)

    raw_data_filepath = os.path.join(storage_path, dataset_filename)
    if not os.path.isfile(raw_data_filepath):
        print("Dataset not found. Downloading.")
        wget.download(dataset_url, raw_data_filepath)
    else:
        print("Dataset found.")
    return raw_data_filepath

def unpack_dataset(storage_path, packed_name, unpacked_name):
    packed_filepath   = os.path.join(storage_path, packed_name)
    unpacked_filepath = os.path.join(storage_path, unpacked_name)

    if not os.path.isfile(unpacked_filepath):
        print("Unpacked dataset not found. Unpacking.")
        with gzip.open(packed_filepath, 'rb') as f_in:
            with open(unpacked_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print("Unpacked dataset found.")

    return unpacked_filepath
