#!/usr/bin/env bash
# Clinical-breakthrough suite on frozen Stage-1 EEG HNF.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CKPT="${CKPT:-outputs/eeg/adftd_hnf_stage1/best.pt}"
OUT="${OUT:-outputs/eeg/clinical_breakthrough_v1}"
DEVICE="${DEVICE:-}"

if [[ -z "$DEVICE" ]]; then
  if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    DEVICE=cuda
  else
    DEVICE=cpu
  fi
fi

mkdir -p logs "$OUT"
echo "[eeg-clinical] ckpt=$CKPT device=$DEVICE → $OUT"
python -u tools/run_eeg_clinical_suite.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_clinical_suite.log

echo "done → $OUT/CLINICAL_REPORT.md"
