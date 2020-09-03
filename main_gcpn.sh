#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

PYARGS=""
PYARGS="$PYARGS --name default"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --artifact_path log_dirs"
PYARGS="$PYARGS --surrogate_model_url https://portal.nersc.gov/project/m3623/docking_score_models/NPRBD_6VYO_BC_1_F_neg_exp1/predict_logp/best_model.pth"

python src/main_gcpn.py $PYARGS
