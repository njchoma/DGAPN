#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dgapn-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name DGAPN_parallel_8_noemb_3d_iota"
PYARGS="$PYARGS --run_id 000"
PYARGS="$PYARGS --nb_procs 8"
# PYARGS="$PYARGS --use_cpu"
PYARGS="$PYARGS --gpu 0"
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgapn"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset NSP15_6W01_A_3_H.negonly_unique_30k.csv"
PYARGS="$PYARGS --reward_type dock" # options: logp, plogp, dock

# PYARGS="$PYARGS --embed_model_path /path/to/trained/sgat.pt"
# PYARGS="$PYARGS --emb_nb_shared 3"
PYARGS="$PYARGS --gnn_nb_layers 3"
PYARGS="$PYARGS --iota 0.1"
PYARGS="$PYARGS --use_3d"

python src/main_train.py $PYARGS
