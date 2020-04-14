#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

PYARGS=""
PYARGS="$PYARGS --name layers_3_with_act"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol"

python src/main.py $PYARGS
