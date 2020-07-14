#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

PYARGS=""
PYARGS="$PYARGS --name default"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol/smiles_scores_ab1f.csv"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol"

python src/main_predict_dock_score.py $PYARGS
