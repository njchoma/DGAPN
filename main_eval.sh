#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env_170_161

PYARGS=""
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
# PYARGS="$PYARGS --data_path $DATA/exaLearnMol/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol/gcpn"
PYARGS="$PYARGS --surrogate_guide_path $ARTIFACTS/exaLearnMol/split_2/predict_logp/best_model.pth"
PYARGS="$PYARGS --surrogate_eval_path  $ARTIFACTS/exaLearnMol/split_2/predict_logp/best_model.pth"
PYARGS="$PYARGS --gcpn_path $ARTIFACTS/exaLearnMol/gcpn_split_1/00301_gcpn.pth"

python src/main_evaluate.py $PYARGS
