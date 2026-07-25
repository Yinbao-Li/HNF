#!/usr/bin/env bash
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
mkdir -p logs outputs/pattern_library_run28 outputs/pattern_routed_eval

CKPT="${CKPT:-outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt}"

echo "[1/2] build pattern library"
/home/bob/anaconda3/bin/python -u tools/build_pattern_library.py \
  --checkpoint "$CKPT" \
  --output-dir outputs/pattern_library_run28 \
  --max-event 2000 --max-noise 1000 --k 6 \
  --device cuda 2>&1 | tee logs/build_pattern_library.log

echo "[2/2] eval routed vs dense + feedback"
/home/bob/anaconda3/bin/python -u tools/eval_pattern_routed_picking.py \
  --checkpoint "$CKPT" \
  --library outputs/pattern_library_run28/pattern_library.json \
  --output-dir outputs/pattern_routed_eval \
  --max-event 500 --max-noise 200 \
  --feedback --device cuda 2>&1 | tee logs/eval_pattern_routed.log

echo "done. see outputs/pattern_library_run28/ and outputs/pattern_routed_eval/report.json"
