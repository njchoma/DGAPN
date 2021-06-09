#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-mol-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name predict_sGAT_3D"
PYARGS="$PYARGS --gpu 0"
PYARGS="$PYARGS --data_path $DATA/src/dataset/NSP15_6W01_A_3_H.negonly_unique_30k.csv" # zinc_plogp_sorted.csv
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --use_3d"

python src/main.py $PYARGS