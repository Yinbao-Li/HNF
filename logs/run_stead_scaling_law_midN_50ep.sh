#!/bin/bash
# Mid/low-N from-scratch sweep @ 50 epochs.
# Separate output root so 12ep/30ep results under outputs/stead_scaling_law/ are kept.
set -euo pipefail
cd /home/bob/TRELLIS/HNF
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-outputs/stead_scaling_law_50ep}"
N_LIST="${N_LIST:-50,100,150,200,300,500,1000}"
MODELS="${MODELS:-hnf,phasenet,eqtransformer}"
EPOCHS_HNF="${EPOCHS_HNF:-50}"
EPOCHS_SB="${EPOCHS_SB:-50}"

mkdir -p logs "$OUT_ROOT" docs/figures/stead

echo "[scaling-50ep] N=$N_LIST models=$MODELS epochs=$EPOCHS_HNF out=$OUT_ROOT"
/home/bob/anaconda3/bin/python -u tools/run_stead_scaling_law.py \
  --output-root "$OUT_ROOT" \
  --n-events "$N_LIST" \
  --models "$MODELS" \
  --epochs-hnf "$EPOCHS_HNF" \
  --epochs-sb "$EPOCHS_SB" \
  --device cuda \
  --skip-existing \
  2>&1 | tee -a logs/stead_scaling_law_midN_50ep.log

/home/bob/anaconda3/bin/python -u tools/analyze_stead_scaling_law.py \
  --root "$OUT_ROOT" \
  --out-fig docs/figures/stead/stead_journal_scaling_law_50ep \
  2>&1 | tee -a logs/stead_scaling_law_midN_50ep.log

echo "[scaling-50ep] done"
