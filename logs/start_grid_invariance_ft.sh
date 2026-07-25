#!/usr/bin/env bash
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
OUT=outputs/run29/29_grid_invariance_ft_v2
mkdir -p logs "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# stop any previous grid-invariance FT
for pat in 'run29_grid_invariance' '29_grid_invariance_ft'; do
  pkill -TERM -f "$pat" 2>/dev/null || true
done
sleep 3
for pat in 'run29_grid_invariance' '29_grid_invariance_ft'; do
  pkill -KILL -f "$pat" 2>/dev/null || true
done
sleep 1

nohup /home/bob/anaconda3/bin/python -u scripts/experiments/run29_grid_invariance_ft.py \
  --device cuda \
  --output-dir "$OUT" \
  > logs/run29_grid_invariance_ft_v2.nohup.log 2>&1 &
echo $! > logs/run29_grid_invariance_ft_v2.pid
echo "started pid=$(cat logs/run29_grid_invariance_ft_v2.pid)"
echo "tail -f $OUT/train.log"
