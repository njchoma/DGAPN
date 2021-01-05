#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env_111_171_163

PYARGS=""
PYARGS="$PYARGS --name default"
PYARGS="$PYARGS --data_path $DATA/exaLearnMol/NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.csv"
PYARGS="$PYARGS --artifact_path $ARTIFACTS/exaLearnMol"
# PYARGS="$PYARGS --use_3d"

python src/main_predict_dock_score.py $PYARGS
