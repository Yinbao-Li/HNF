#!/usr/bin/env bash
# Phase 2.2: train EQT-aug / PN-aug to 182 ep; save ep 20/40/80/182; eval each.
set -euo pipefail
cd /home/bob/TRELLIS/HNF
PY=${PY:-python}
OUT=results/phase2/2_2_matched_budget
mkdir -p "$OUT"/{eqt,phasenet,eval}
LOG=$OUT/run_train.log
exec > >(tee -a "$LOG") 2>&1

echo "[2.2] start $(date -Is)"
COMMON=(
  --device cuda --epochs 182 --warmup 2 --batch-size 16 --lr 1e-3
  --weight-decay 1e-4 --samples-per-epoch 200000 --max-val 8000
  --num-workers 4 --seed 42 --label-sigma-sec 0.35
  --checkpoint-epochs 20,40,80,182
)

train_one() {
  local model=$1 dir=$2
  echo "[2.2] train $model → $dir $(date -Is)"
  $PY tools/train_sb_stead_aug.py --model "$model" --output-dir "$dir" "${COMMON[@]}"
  # also keep ep0020 copy named for eval loop
  echo "[2.2] done train $model $(date -Is)"
}

# From-scratch (clean 182-ep cosine); do not resume annealed 20-ep last.pt
train_one eqt "$OUT/eqt"
train_one phasenet "$OUT/phasenet"

echo "[2.2] training complete $(date -Is)"
echo "[2.2] launching evals at checkpoints..."
$PY tools/phase2_eval_matched_budget.py --root "$OUT" --device cuda
echo "[2.2] ALL DONE $(date -Is)"
