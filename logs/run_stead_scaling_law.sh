#!/bin/bash
# STEAD scaling-law sweep: HNF / PhaseNet / EQTransformer
# Default caps at N=30k (no 100k/240k). Low-N 30ep: see run_stead_scaling_law_lowN_30ep.sh
set -euo pipefail
cd /home/bob/TRELLIS/HNF
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
mkdir -p logs outputs/stead_scaling_law docs/figures/stead

N_LIST="${N_LIST:-1000,3000,10000,30000}"
MODELS="${MODELS:-hnf,phasenet,eqtransformer}"
EPOCHS_HNF="${EPOCHS_HNF:-12}"
EPOCHS_SB="${EPOCHS_SB:-12}"

echo "[scaling] N=$N_LIST models=$MODELS epochs_hnf=$EPOCHS_HNF epochs_sb=$EPOCHS_SB"
/home/bob/anaconda3/bin/python -u tools/run_stead_scaling_law.py \
  --output-root outputs/stead_scaling_law \
  --n-events "$N_LIST" \
  --models "$MODELS" \
  --epochs-hnf "$EPOCHS_HNF" \
  --epochs-sb "$EPOCHS_SB" \
  --device cuda \
  --skip-existing \
  2>&1 | tee -a logs/stead_scaling_law.log

echo "[scaling] done"
