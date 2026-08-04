#!/bin/bash
# Differentiable Kernel Bank — resume after OOM with lower memory
set -euo pipefail
cd /home/bob/TRELLIS/HNF
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="${OUT:-outputs/kernel_bank_v0}"
mkdir -p logs "$OUT"

CKPT="$OUT/last.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[kernel-bank] missing $CKPT" >&2
  exit 1
fi

echo "[kernel-bank] resume $CKPT → $OUT (bs=2, top_m=2)"
/home/bob/anaconda3/bin/python -u tools/train_stead_picking.py \
  --output-dir "$OUT" \
  --resume "$CKPT" \
  --continue \
  --epochs 30 \
  --batch-size 2 \
  --grad-accum-steps 24 \
  --lr 3e-4 \
  --seq-len 800 \
  --embed-dim 64 \
  --num-shared-layers 2 \
  --num-branch-layers 2 \
  --local-window-sec 15.0 \
  --seed 42 \
  --max-event-train 5000 \
  --max-noise-train 2500 \
  --max-val 2000 \
  --multi-scale \
  --principle huygens_fresnel \
  --obliquity-scale 1.0 \
  --sparse-band \
  --kernel-bank-size 8 \
  --kernel-bank-top-m 2 \
  --kernel-bank-reg-scale 1.0 \
  --rho-sparsity-weight 0.02 \
  --rho-sparsity-radius-sec 1.5 \
  --kernel-phys-prior-weight 0.005 \
  --pick-head-hidden 48 \
  --pick-head-layers 4 \
  --pick-head-kernel 7 \
  --noise-source-dim 16 \
  --no-residual-det-head \
  --enhanced-det-head \
  --noise-cancel \
  --noise-pick-cues \
  --noise-det-pick-split \
  --noise-cancel-weight 0.05 \
  --wrong-peak-loss-weight 0.15 \
  --wrong-peak-radius-sec 0.45 \
  --wrong-peak-margin 0.25 \
  --s-wrong-peak-scale 1.35 \
  --ps-order-loss-weight 0.12 \
  --ps-min-gap-sec 0.1 \
  --post-process-p-before-s \
  --pick-loss-weight 2.8 \
  --pick-pos-weight 28 \
  --p-pick-loss-weight 1.3 \
  --s-pick-loss-weight 1.6 \
  --det-event-weight 2.0 \
  --label-sigma-sec 0.35 \
  --score-mode det_guard \
  --det-score-floor 0.988 \
  --skip-final-test \
  --device cuda \
  2>&1 | tee -a logs/kernel_bank_v0.log

echo "[kernel-bank] done → $OUT"
