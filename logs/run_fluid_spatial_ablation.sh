#!/usr/bin/env bash
# Spatial HNF + rotational sources — vortex ablation (raster vs spatial±rot)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/bob/anaconda3/bin/python}"
EPOCHS="${EPOCHS:-40}"
DEVICE="${DEVICE:-cuda}"
FAMILIES="${FAMILIES:-vortex}"

exec "$PYTHON" scripts/experiments/run_fluid_spatial_ablation.py \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --families "$FAMILIES" \
  "$@"
