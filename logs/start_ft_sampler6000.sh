#!/usr/bin/env bash
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
mkdir -p logs outputs/run28/28_ft_sampler6000_8ep
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 1) stop the old native-6000 probe (still stuck in slow full-test on a known-bad ckpt)
if [[ -f logs/run28_ft_6000_5ep_probe.pid ]]; then
  old=$(cat logs/run28_ft_6000_5ep_probe.pid || true)
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "killing old probe pid=${old} (+children)"
    pkill -TERM -P "${old}" 2>/dev/null || true
    kill -TERM "${old}" 2>/dev/null || true
    sleep 5
    pkill -KILL -P "${old}" 2>/dev/null || true
    kill -KILL "${old}" 2>/dev/null || true
  fi
fi
# belt-and-suspenders: kill any lingering probe trainer
pkill -f 'run28_ft_6000_5ep_probe.py' 2>/dev/null || true
pkill -f 'train_stead_picking.py .*28_ft_6000_5ep_probe' 2>/dev/null || true
sleep 3

# 2) launch grid-aligned sampler FT
nohup /home/bob/anaconda3/bin/python -u scripts/experiments/run28_ft_sampler6000_8ep.py \
  --device cuda \
  > logs/run28_ft_sampler6000_8ep.nohup.log 2>&1 &
echo $! > logs/run28_ft_sampler6000_8ep.pid
echo "started pid=$(cat logs/run28_ft_sampler6000_8ep.pid)"
echo "tail -f outputs/run28/28_ft_sampler6000_8ep/train.log"
