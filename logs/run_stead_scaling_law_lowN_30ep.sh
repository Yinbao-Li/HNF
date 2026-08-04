#!/bin/bash
# Low-N from-scratch sweep: N=50,200,500 × HNF/PN/EQT @ 30 epochs
set -euo pipefail
cd /home/bob/TRELLIS/HNF
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/stead_scaling_law docs/figures/stead

N_LIST="${N_LIST:-50,200,500}"
MODELS="${MODELS:-hnf,phasenet,eqtransformer}"
EPOCHS_HNF="${EPOCHS_HNF:-30}"
EPOCHS_SB="${EPOCHS_SB:-30}"

echo "[scaling-lowN] N=$N_LIST models=$MODELS epochs_hnf=$EPOCHS_HNF epochs_sb=$EPOCHS_SB"
/home/bob/anaconda3/bin/python -u tools/run_stead_scaling_law.py \
  --output-root outputs/stead_scaling_law \
  --n-events "$N_LIST" \
  --models "$MODELS" \
  --epochs-hnf "$EPOCHS_HNF" \
  --epochs-sb "$EPOCHS_SB" \
  --device cuda \
  --skip-existing \
  2>&1 | tee -a logs/stead_scaling_law_lowN_30ep.log

echo "[scaling-lowN] done"
