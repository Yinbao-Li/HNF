#!/usr/bin/env bash
# Train EEG-native HNF v5 (regional + δ/θ/α + segment pool), then clinical suite.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# v5 recipe: regional + δ + segment pool; mild MMSE; NO subject over-balance
# (v2/v4 hard-balance / disease-boost collapsed AD↔FTD).
OUT="${OUT:-outputs/eeg/adftd_hnf_native_v5}"
CLIN_OUT="${CLIN_OUT:-outputs/eeg/clinical_breakthrough_native_v5}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
MMSE_W="${MMSE_W:-0.1}"

mkdir -p logs "$OUT" "$CLIN_OUT"
echo "[eeg-native] train → $OUT  device=$DEVICE epochs=$EPOCHS lr=$LR mmse_w=$MMSE_W"
python -u tools/train_eeg_native.py \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --mmse-weight "$MMSE_W" \
  --no-subject-balanced \
  --arch-tag "eeg_hnf_native_v5" \
  --no-synthetic \
  2>&1 | tee logs/eeg_native_v5_train.log

echo "[eeg-native] clinical suite → $CLIN_OUT"
python -u tools/run_eeg_clinical_suite.py \
  --checkpoint "$OUT/best.pt" \
  --output-dir "$CLIN_OUT" \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_native_v5_clinical.log

echo "done → $OUT + $CLIN_OUT"
