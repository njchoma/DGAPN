#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-new

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name crem_simplified_parallel_16"
PYARGS="$PYARGS --gpu 3"
PYARGS="$PYARGS --nb_procs 16"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset_path $DATA/src/dataset/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --surrogate_model_path $DATA/artifact/surrogate_model.pth"

python src/main_gcpn.py $PYARGS
