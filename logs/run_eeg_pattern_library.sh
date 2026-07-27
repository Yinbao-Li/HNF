#!/usr/bin/env bash
# Induce EEG pattern library from a frozen clinical ckpt, then route-eval.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CKPT="${CKPT:-outputs/eeg/adftd_hnf_native_v3/best.pt}"
OUT="${OUT:-outputs/eeg/pattern_library_native_v3_tight}"
DEVICE="${DEVICE:-cuda}"
K="${K:-6}"

mkdir -p logs "$OUT"
echo "[eeg-pattern-lib] ckpt=$CKPT → $OUT  k=$K (tight+calibrate+online counters)"
python -u tools/build_eeg_pattern_library.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --k "$K" \
  --no-synthetic \
  --online-update \
  2>&1 | tee logs/eeg_pattern_library.log

echo "done → $OUT"
