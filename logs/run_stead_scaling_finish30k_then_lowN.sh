#!/bin/bash
# 1) Finish remaining N=30k jobs @12ep (skip-existing).
# 2) Run N=50,200,500 @30ep for HNF / PhaseNet / EQTransformer.
set -euo pipefail
cd /home/bob/TRELLIS/HNF
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/stead_scaling_law docs/figures/stead

PY=/home/bob/anaconda3/bin/python

echo "[stage1] finish N=30000 @12ep (skip completed)"
N_LIST=30000 EPOCHS_HNF=12 EPOCHS_SB=12 bash logs/run_stead_scaling_law.sh

echo "[stage2] low-N N=50,200,500 @30ep"
N_LIST=50,200,500 EPOCHS_HNF=30 EPOCHS_SB=30 bash logs/run_stead_scaling_law_lowN_30ep.sh

echo "[all] analyze"
"$PY" -u tools/analyze_stead_scaling_law.py --root outputs/stead_scaling_law \
  2>&1 | tee -a logs/stead_scaling_law.log

echo "[all] done"
