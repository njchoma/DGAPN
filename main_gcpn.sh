#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate basic_dock3

PYARGS=""
PYARGS="$PYARGS --name default"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol/0513_gcpn_basic_dock"
PYARGS="$PYARGS --surrogate_model_path $ARTIFACTS/exaLearnMol/default/predict_logp/best_model.pth"
PYARGS="$PYARGS --surrogate_reward_timestep_delay 0"
PYARGS="$PYARGS --prob_redux_factor 0.99"

python src/main_gcpn.py $PYARGS
