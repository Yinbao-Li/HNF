#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/bob/anaconda3/bin/python}"
exec "$PYTHON" scripts/experiments/run_fluid_3d4d_suite.py --device "${DEVICE:-cuda}" --epochs "${EPOCHS:-40}" "$@"
