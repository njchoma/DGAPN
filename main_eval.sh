#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-new

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name dgapn_eval"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset_path $DATA/src/dataset/NSP15_6W01_A_3_H.negonly_unique.csv" # zinc_plogp_sorted.csv
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgapn"
PYARGS="$PYARGS --policy_path /clusterfs/csdata/data/gcpn/dgapn_policy/running_dgapn2.pth"
PYARGS="$PYARGS --nb_test 1000"
PYARGS="$PYARGS --reward_type dock"
# PYARGS="$PYARGS --greedy"

python src/main_evaluate.py $PYARGS
