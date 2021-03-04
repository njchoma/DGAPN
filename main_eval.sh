#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env_170_161

PYARGS=""
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
# PYARGS="$PYARGS --data_path $DATA/exaLearnMol/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol/gcpn"
PYARGS="$PYARGS --surrogate_model_path $ARTIFACTS/exaLearnMol/test_2/predict_logp/best_model.pth"
PYARGS="$PYARGS --gcpn_path $ARTIFACTS/exaLearnMol/gcpn/00101_gcpn.pth"

python src/main_evaluate.py $PYARGS
