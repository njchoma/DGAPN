#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env_name

PYARGS=""
PYARGS="$PYARGS --data_path $DATA/exaLearnMol"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol"

python src/main.py $PYARGS
