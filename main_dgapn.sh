#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-mol-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name DGAPN_parallel_8_noemb_3d_iota"
PYARGS="$PYARGS --gpu 0"
PYARGS="$PYARGS --nb_procs 8"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset_path $DATA/src/dataset/NSP15_6W01_A_3_H.negonly_unique_30k.csv" # zinc_plogp_sorted.csv
PYARGS="$PYARGS --artifact_path $DATA/artifact/dgapn"
PYARGS="$PYARGS --reward_type dock"
PYARGS="$PYARGS --adt_tmp_dir 001"
# PYARGS="$PYARGS --embed_model_path /path/to/trained/embed_model.pt"
# PYARGS="$PYARGS --emb_nb_shared 3"
PYARGS="$PYARGS --gnn_nb_layers 3"
PYARGS="$PYARGS --iota 0.08"
PYARGS="$PYARGS --use_3d"

python src/main_dgapn.py $PYARGS
