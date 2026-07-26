#!/usr/bin/env bash
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
OUT=outputs/run30/30_grid_highhz_ft
mkdir -p logs "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for pat in 'run30_grid_highhz' '30_grid_highhz_ft' 'run29_grid_invariance'; do
  pkill -TERM -f "$pat" 2>/dev/null || true
done
sleep 2
for pat in 'run30_grid_highhz' '30_grid_highhz_ft'; do
  pkill -KILL -f "$pat" 2>/dev/null || true
done
sleep 1

nohup /home/bob/anaconda3/bin/python -u scripts/experiments/run30_grid_highhz_ft.py \
  --device cuda \
  --output-dir "$OUT" \
  > logs/run30_grid_highhz_ft.nohup.log 2>&1 &
echo $! > logs/run30_grid_highhz_ft.pid
echo "started pid=$(cat logs/run30_grid_highhz_ft.pid)"
echo "tail -f $OUT/train.log"
