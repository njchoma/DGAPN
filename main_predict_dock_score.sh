#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-new

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name predict_dock_default"
PYARGS="$PYARGS --data_path $DATA/src/dataset/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact/gcpn"
# PYARGS="$PYARGS --use_3d"

python src/main_predict_dock_score.py $PYARGS
