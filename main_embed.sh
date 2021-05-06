#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-new

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name predict_dock_DGAPN_A3H_sub_3D"
PYARGS="$PYARGS --gpu 3"
PYARGS="$PYARGS --data_path $DATA/src/dataset/NSP15_6W01_A_3_H.negonly_unique_30k.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgapn"
PYARGS="$PYARGS --use_3d"

python src/main_embed.py $PYARGS
