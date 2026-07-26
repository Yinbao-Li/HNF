#!/usr/bin/env bash
# Pattern library + routed eval. Default: run29 grid-invariant best, coarse=400
# (inside the FT grid range). Override with CKPT=... COARSE_LEN=...
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF

CKPT="${CKPT:-outputs/run29/29_grid_invariance_ft_v2/best.pt}"
COARSE_LEN="${COARSE_LEN:-400}"
LIB_DIR="${LIB_DIR:-outputs/pattern_library_run29}"
EVAL_DIR="${EVAL_DIR:-outputs/pattern_routed_eval_run29}"
mkdir -p logs "$LIB_DIR" "$EVAL_DIR"

echo "[1/3] build pattern library  ckpt=$CKPT  coarse=$COARSE_LEN"
/home/bob/anaconda3/bin/python -u tools/build_pattern_library.py \
  --checkpoint "$CKPT" \
  --output-dir "$LIB_DIR" \
  --max-event 2000 --max-noise 1000 --k 6 \
  --coarse-len "$COARSE_LEN" \
  --device cuda 2>&1 | tee logs/build_pattern_library_run29.log

echo "[2/3] eval routed vs dense (feedback counts only; no centre drift)"
/home/bob/anaconda3/bin/python -u tools/eval_pattern_routed_picking.py \
  --checkpoint "$CKPT" \
  --library "$LIB_DIR/pattern_library.json" \
  --output-dir "$EVAL_DIR" \
  --max-event 500 --max-noise 200 \
  --feedback --device cuda 2>&1 | tee logs/eval_pattern_routed_run29.log

echo "[3/3] eval det-gate (native grid, no coarse downsample)"
/home/bob/anaconda3/bin/python -u tools/eval_pattern_routed_picking.py \
  --checkpoint "$CKPT" \
  --library "$LIB_DIR/pattern_library.json" \
  --output-dir "${EVAL_DIR}_detgate" \
  --max-event 500 --max-noise 200 \
  --det-gate --gate-threshold 0.35 \
  --device cuda 2>&1 | tee logs/eval_pattern_detgate_run29.log

echo "done."
echo "  library: $LIB_DIR/"
echo "  routed:  $EVAL_DIR/report.json"
echo "  detgate: ${EVAL_DIR}_detgate/report.json"
