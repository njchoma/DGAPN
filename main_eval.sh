#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-new

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name dgapn_eval"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset_path $DATA/src/dataset/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgapn"
PYARGS="$PYARGS --surrogate_guide_path $DATA/artifact/surrogate_model.pth"
PYARGS="$PYARGS --surrogate_eval_path $DATA/artifact/surrogate_model.pth"
PYARGS="$PYARGS --model_path $DATA/artifact/dgapn.pth"
# PYARGS="$PYARGS --greedy"

python src/main_evaluate.py $PYARGS
