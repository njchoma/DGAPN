#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate dgapn-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name DGAPN_eval_1000_0.4ts4"
PYARGS="$PYARGS --run_id 022"
PYARGS="$PYARGS --gpu 2" # PYARGS="$PYARGS --use_cpu"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --warm_start_dataset NSP15_6W01_A_3_H.negonly_unique_100.csv"
PYARGS="$PYARGS --artifact_path $DATA/artifact/constrain_opt"
PYARGS="$PYARGS --reward_type dock" # options: logp, plogp, dock

PYARGS="$PYARGS --model_path /global/home/users/yulunwu/exaLearnMol/artifact/constrain_opt/saves/delta0.4_ts4_2021.10.13_14:22:55/02000_dgapn.pt" # PYARGS="$PYARGS --greedy"
PYARGS="$PYARGS --nb_test 1000"
PYARGS="$PYARGS --max_rollout 4"

python src/main_evaluate.py $PYARGS
