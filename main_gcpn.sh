#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name NSP15_6W01_A2F_molpher"
PYARGS="$PYARGS --data_path $DATA"
PYARGS="$PYARGS --artifact_path $DATA/gcpn"
PYARGS="$PYARGS --use_surrogate"
PYARGS="$PYARGS --use_crem"
PYARGS="$PYARGS --gpu 2"
PYARGS="$PYARGS --surrogate_model_url https://portal.nersc.gov/project/m3623/docking_score_models/NSP15_6W01_A_2_F_neg_exp/predict_logp/best_model.pth"
PYARGS="$PYARGS --is_conditional 1"
PYARGS="$PYARGS --conditional /global/home/users/adchen/MD/2col/negative_only/neg_only_NSP15_6W01_A_2_F.Orderable_zinc_db_enaHLL.2col.csv"
PYARGS="$PYARGS --surrogate_reward_timestep_delay 10"

python src/main_gcpn.py $PYARGS
