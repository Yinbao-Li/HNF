#!/usr/bin/env bash
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
mkdir -p logs outputs/run28/28_ft_linear6000_8ep_v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# stop any previous linear FT
for pat in 'run28_ft_linear6000' '28_ft_linear6000_8ep'; do
  pkill -TERM -f "$pat" 2>/dev/null || true
done
sleep 3
for pat in 'run28_ft_linear6000' '28_ft_linear6000_8ep'; do
  pkill -KILL -f "$pat" 2>/dev/null || true
done
sleep 1

nohup /home/bob/anaconda3/bin/python -u scripts/experiments/run28_ft_linear6000_8ep.py \
  --device cuda \
  --output-dir outputs/run28/28_ft_linear6000_8ep_v2 \
  > logs/run28_ft_linear6000_8ep_v2.nohup.log 2>&1 &
echo $! > logs/run28_ft_linear6000_8ep_v2.pid
echo "started pid=$(cat logs/run28_ft_linear6000_8ep_v2.pid)"
echo "tail -f outputs/run28/28_ft_linear6000_8ep_v2/train.log"
