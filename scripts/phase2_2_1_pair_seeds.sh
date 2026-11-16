#!/usr/bin/env bash
# Phase 2.1 reduced: 20-ep Huygens vs free pair × seeds 42,43,44.
# Seed 42 already exists under outputs/stead_c2f_pair_{huygens,free}; reuse.
# New runs: seeds 43,44 × {huygens,free}.
set -euo pipefail
cd /home/bob/TRELLIS/HNF
PY=${PY:-python}
OUT=results/phase2/2_1_pair_seeds
mkdir -p "$OUT"
LOG=$OUT/run_train.log
exec > >(tee -a "$LOG") 2>&1

echo "[2.1] start $(date -Is)"

COMMON=(
  --device cuda --epochs 20 --warmup 2 --batch-size 16 --lr 5e-4
  --samples-per-epoch 200000 --max-event-train 10000000 --max-noise-train 10000000
  --max-val 8000 --num-workers 4 --amp --amp-dtype bf16 --fresh-augment --val-shift
  --skip-final-test --pick-pos-weight 1 --pick-threshold 0.3
  --aug-shift-prob 0.5 --aug-second-event-prob 0.3 --aug-real-noise-prob 0.5
  --aug-real-noise-snr-min 0 --aug-real-noise-snr-max 20 --aug-gauss-prob 0.3
  --aug-response-prob 0.3 --aug-gap-prob 0.2 --aug-channel-drop-prob 0.1
  --aug-phase-absent-prob 0.3
  --fusion propose_refine --wave-input field --skip-bottleneck 8 --time-head-drop 0.15
  --wave-sup-weight 0.5 --noise-max-weight 0.1 --refine-sup-weight 1.0
  --refine-label-sigma-sec 0.1 --refine-pool-sec 0.2
  --label-sigma-sec 0.35
)

# Link seed 42 existing
for arm in huygens free; do
  d=$OUT/${arm}_seed42
  mkdir -p "$d"
  if [[ ! -e $d/last.pt ]]; then
    ln -sfn "$(pwd)/outputs/stead_c2f_pair_${arm}/last.pt" "$d/last.pt"
    ln -sfn "$(pwd)/outputs/stead_c2f_pair_${arm}/history.json" "$d/history.json"
    ln -sfn "$(pwd)/outputs/stead_c2f_pair_${arm}/args.json" "$d/args.json"
  fi
done

for seed in 43 44; do
  for arm in huygens free; do
    prop=$arm
    d=$OUT/${arm}_seed${seed}
    mkdir -p "$d"
    if [[ -f $d/last.pt ]]; then
      echo "[2.1] skip existing $d"
      continue
    fi
    echo "[2.1] train $arm seed=$seed → $d $(date -Is)"
    $PY tools/train_stead_c2f_picking.py --output-dir "$d" --propagation "$prop" --seed "$seed" \
      "${COMMON[@]}" > "$d/run.log" 2>&1
    echo "[2.1] done $arm seed=$seed $(date -Is)"
  done
done

echo "[2.1] training complete; running eval aggregate $(date -Is)"
$PY tools/phase2_eval_pair_seeds.py --root "$OUT" --device cuda
echo "[2.1] ALL DONE $(date -Is)"
