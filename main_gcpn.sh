#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate gcpn3

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name NSP15_6W01_A1F_uncharged_kernel_dist"
PYARGS="$PYARGS --data_path $DATA"
PYARGS="$PYARGS --artifact_path $DATA/gcpn"
PYARGS="$PYARGS --use_surrogate"
PYARGS="$PYARGS --use_crem"
PYARGS="$PYARGS --gpu 3"
PYARGS="$PYARGS --surrogate_model_url "
PYARGS="$PYARGS --is_conditional 1"
PYARGS="$PYARGS --conditional /global/home/users/adchen/MD/2col/uncharged_unique/uncharged_NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.2col.csv"
PYARGS="$PYARGS --surrogate_reward_timestep_delay 10"
PYARGS="$PYARGS --stochastic_kernel"
PYARGS="$PYARGS --heads_g 2"

python src/main_gcpn.py $PYARGS
