#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name NSP15_6W01_A2F_master"
PYARGS="$PYARGS --data_path $DATA"
PYARGS="$PYARGS --artifact_path $DATA/gcpn"
PYARGS="$PYARGS --gpu 3"
PYARGS="$PYARGS --use_surrogate"
PYARGS="$PYARGS --surrogate_model_url https://portal.nersc.gov/project/m3623/docking_score_models/NSP15_6W01_A_2_F_neg_exp/predict_logp/best_model.pth"
PYARGS="$PYARGS --surrogate_reward_episode_delay 10"
PYARGS="$PYARGS --is_conditional 1"
PYARGS="$PYARGS --conditional /global/home/users/adchen/MD/2col/negative_only/neg_only_NSP15_6W01_A_2_F.Orderable_zinc_db_enaHLL.2col.csv"
PYARGS="$PYARGS --adversarial_reward_episode_delay 20"

python src/main_gcpn.py $PYARGS
