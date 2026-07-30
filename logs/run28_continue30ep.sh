#!/usr/bin/env bash
# run28 STEAD continue: ep50 last.pt → +30 epochs (80 total)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOG="${ROOT}/logs/run28_continue30ep.nohup.log"
PID="${ROOT}/logs/run28_continue30ep.pid"
OUT="${ROOT}/outputs/run28/28_ms_fresnel_phys_80ep_local"

mkdir -p logs outputs/run28

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "[run28-continue30] already running pid=$(cat "$PID")"
  exit 0
fi

nohup python scripts/experiments/run28_stead_continue30ep.py --device cuda \
  > "$LOG" 2>&1 &
echo $! > "$PID"
echo "[run28-continue30] pid=$(cat "$PID") log=$LOG out=$OUT"
