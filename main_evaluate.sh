#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-mol-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name dgapn_eval"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset_path $DATA/src/dataset/NSP15_6W01_A_3_H.negonly_unique.csv" # zinc_plogp_sorted.csv
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgapn"
PYARGS="$PYARGS --policy_path /path/to/trained/dgapn_policy.pth"
PYARGS="$PYARGS --nb_test 1000"
PYARGS="$PYARGS --reward_type dock"
# PYARGS="$PYARGS --greedy"

python src/main_evaluate.py $PYARGS
