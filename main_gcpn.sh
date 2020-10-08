#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

PYARGS=""
PYARGS="$PYARGS --name Ep0_rewFacNone"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol/gcpn"
PYARGS="$PYARGS --surrogate_reward"
PYARGS="$PYARGS --surrogate_reward_timestep_delay 10"
PYARGS="$PYARGS --surrogate_model_path $ARTIFACTS/exaLearnMol/rdk3_dock_nsp15_6w01_a_1_with_exp/predict_logp/best_model.pth"

python src/main_gcpn.py $PYARGS
