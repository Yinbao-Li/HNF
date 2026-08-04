#!/usr/bin/env bash
# Stable spatial HNF suite (vortex → all families → RACLETTE)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/bob/anaconda3/bin/python}"
DEVICE="${DEVICE:-cuda}"

exec "$PYTHON" scripts/experiments/run_fluid_spatial_suite.py \
  --device "$DEVICE" \
  --epochs-vortex 50 \
  --epochs-all 50 \
  --epochs-raclette 40 \
  "$@"
