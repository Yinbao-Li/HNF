#!/usr/bin/env bash
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
mkdir -p logs outputs/run28/28_ft_6000_5ep_probe
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# kill previous probe if still around
if [[ -f logs/run28_ft_6000_5ep_probe.pid ]]; then
  old=$(cat logs/run28_ft_6000_5ep_probe.pid || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    kill "${old}" || true
    sleep 2
  fi
fi

nohup /home/bob/anaconda3/bin/python -u scripts/experiments/run28_ft_6000_5ep_probe.py \
  --device cuda \
  > logs/run28_ft_6000_5ep_probe.nohup.log 2>&1 &
echo $! > logs/run28_ft_6000_5ep_probe.pid
echo "started pid=$(cat logs/run28_ft_6000_5ep_probe.pid)"
echo "tail -f outputs/run28/28_ft_6000_5ep_probe/train.log"
