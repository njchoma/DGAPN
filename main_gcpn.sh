#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env_170_161

PYARGS=""
PYARGS="$PYARGS --name 0225_index_fix"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --warm_start_dataset_path $DATA/exaLearnMol/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol/gcpn_split_1"
PYARGS="$PYARGS --surrogate_model_path $ARTIFACTS/exaLearnMol/split_1/predict_logp/best_model.pth"

python src/main_gcpn.py $PYARGS
