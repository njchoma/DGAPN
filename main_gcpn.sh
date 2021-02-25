#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name NSP15_6W01_A2F_uncharged_crem_adversarial"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --gpu 2"
PYARGS="$PYARGS --use_crem"
PYARGS="$PYARGS --is_conditional 1"
PYARGS="$PYARGS --conditional src/dataset/uncharged_NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.2col.csv"
PYARGS="$PYARGS --use_surrogate"
PYARGS="$PYARGS --surrogate_model_url https://portal.nersc.gov/project/m3623/docking_score_models/NSP15_6W01_A_1_F_uncharged_upsamp/predict_logp/best_model.pth"
PYARGS="$PYARGS --surrogate_reward_episode_delay 10"
PYARGS="$PYARGS --use_adversarial"
PYARGS="$PYARGS --adversarial_reward_episode_cutoff 1000"

python src/main_gcpn.py $PYARGS
