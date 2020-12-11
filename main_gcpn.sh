#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env_170_161

PYARGS=""
PYARGS="$PYARGS --name Ep0_rewFacNone"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol/gcpn"
PYARGS="$PYARGS --surrogate_model_path $ARTIFACTS/exaLearnMol/default/predict_logp/best_model.pth"
PYARGS="$PYARGS --surrogate_reward_timestep_delay 10"

python src/main_gcpn.py $PYARGS
