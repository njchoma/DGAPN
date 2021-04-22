#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gcpn

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --warm_start_dataset_path /Users/ADChen/Downloads/MD/2col/NSP15_6W01_A_1_F_combined_sorted_negonly_unique.csv"
PYARGS="$PYARGS --artifact_path ."
PYARGS="$PYARGS --surrogate_guide_path artifact/surrogate_model.pth"
PYARGS="$PYARGS --surrogate_eval_path  artifact/surrogate_model.pth"
PYARGS="$PYARGS --gcpn_path /Users/ADChen/Downloads/saves/crem_parallel_GPU_8_2021.04.12_16:27:07/gcpn_actor_test.pth"

python src/main_evaluate.py $PYARGS
