#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dgapn-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name DGAPN_eval_1000"
PYARGS="$PYARGS --run_id 010"
PYARGS="$PYARGS --gpu 0" # PYARGS="$PYARGS --use_cpu"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset NSP15_6W01_A_3_H.negonly_unique_300k.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact/plogp"
PYARGS="$PYARGS --reward_type plogp" # options: logp, plogp, dock

PYARGS="$PYARGS --model_path /global/home/users/yulunwu/exaLearnMol/artifact/dgapn/saves/DGAPN_parallel_8_plogp_2021.10.02_23:14:24/03724_dgapn.pt" # PYARGS="$PYARGS --greedy"
PYARGS="$PYARGS --nb_test 1000"

python src/main_evaluate.py $PYARGS
