#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gcpndock

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name crem_parallel_grow_GPU_8"
PYARGS="$PYARGS --gpu 1"
PYARGS="$PYARGS --nb_procs 8"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset_path /global/home/users/adchen/MD/2col/uncharged_unique/NSP15_6W01_A_1_F_combined_sorted_negonly.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --surrogate_model_path $DATA/artifact/best_model.pth"

python src/main_gcpn.py $PYARGS
