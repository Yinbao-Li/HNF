#!/usr/bin/env bash
# Reproduce frozen macro-head Zhizi inversion route.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
HEAD="${PHYSICS_HEAD:-outputs/zhizi_inversion_bridge_macro/best_physics_head.pt}"
CKPT="${CHECKPOINT:-outputs/run20/20_wrongpeak_sharp/best.pt}"

echo "[1/2] Route A2 synthetic (32 events)..."
PYTHONUNBUFFERED=1 python run_route_a2_waveform.py \
  --checkpoint "$CKPT" \
  --physics-head "$HEAD" \
  --head-mode macro \
  --n-test 32 \
  --fwi-steps 60 \
  --device "$DEVICE" \
  --output-dir outputs/route_a2_waveform_macro_repro

echo "[2/2] STEAD inv05-real macro comparison..."
PYTHONUNBUFFERED=1 python scripts/inversion/run_zhizi_inv05_real.py \
  --checkpoint "$CKPT" \
  --physics-head "$HEAD" \
  --head-mode macro \
  --obs-fallback \
  --max-events 24 \
  --device "$DEVICE" \
  --output-dir outputs/zhizi_inv05_real_macro

echo "[done] see outputs/route_a2_waveform_macro_repro and outputs/zhizi_inv05_real_macro"
