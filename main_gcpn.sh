#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate my-rdkit-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name NSP15_6W01_A1F_uncharged_kernel_dist"
PYARGS="$PYARGS --data_path $DATA/src/dataset"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --use_surrogate"
PYARGS="$PYARGS --use_crem"
PYARGS="$PYARGS --gpu 3"
PYARGS="$PYARGS --surrogate_model_url https://portal.nersc.gov/project/m3623/docking_score_models/NSP15_6W01_A_1_F_uncharged_upsamp/predict_logp/best_model.pth"
PYARGS="$PYARGS --is_conditional 1"
PYARGS="$PYARGS --conditional src/dataset/uncharged_NSP15_6W01_A_1_F.Orderable_zinc_db_enaHLL.2col.csv"
PYARGS="$PYARGS --surrogate_reward_timestep_delay 10"
PYARGS="$PYARGS --stochastic_kernel"
PYARGS="$PYARGS --reset_projections"
PYARGS="$PYARGS --layer_num_g 4"

python src/main_gcpn.py $PYARGS
