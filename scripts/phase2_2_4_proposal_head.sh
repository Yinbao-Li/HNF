#!/usr/bin/env bash
# Phase 2.4: proposal-head-only FT + pre-registered SNR-aware proposal score.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/stead_c2f_2_4_proposal_head
LOG=results/phase2/2_4_low_snr/train_proposal_head.log
mkdir -p "$OUT" results/phase2/2_4_low_snr
echo "[2.4] start $(date -Iseconds)" | tee -a "$LOG"
python -u tools/train_stead_c2f_picking.py \
  --output-dir "$OUT" \
  --resume outputs/stead_c2f_imp_RSP/last.pt \
  --fusion propose_refine --wave-input field \
  --refine-max 4.0 --refine-pool-sec 0.2 --gate-floor 0.0 \
  --time-head-drop 0.15 --skip-bottleneck 8 \
  --principle huygens_fresnel --sparse-band \
  --snr-aware-proposal \
  --snr-mad-floor 0.05 --snr-z-floor 2.0 --snr-boost 0.84729786 --snr-soft-k 4.0 \
  --proposal-head-only \
  --epochs 6 --batch-size 16 --lr 1e-4 --warmup 0 --lr-schedule cosine \
  --samples-per-epoch 100000 --fresh-augment --amp --amp-dtype bf16 \
  --wave-sup-weight 1.0 --wave-pos-weight 3.0 \
  --refine-sup-weight 1.0 --refine-label-sigma-sec 0.1 \
  --pick-pos-weight 1.0 --pick-threshold 0.3 --pick-tolerance-sec 0.5 \
  --label-sigma-sec 0.35 --late-sigma-ratio 0.4 --manual-weight 2.0 \
  --noise-loss-weight 0.15 --noise-max-weight 0.1 \
  --aug-shift-prob 0.5 --aug-shift-p-min 1.0 --aug-shift-p-max 50.0 \
  --aug-channel-drop-prob 0.1 --aug-gauss-prob 0.3 --aug-gauss-snr-min 5 --aug-gauss-snr-max 20 \
  --aug-second-event-prob 0.3 \
  --aug-real-noise-prob 0.8 --aug-real-noise-snr-min -5 --aug-real-noise-snr-max 10 \
  --aug-response-prob 0.3 --aug-gap-prob 0.2 --aug-phase-absent-prob 0.3 \
  --aug-lowfreq-prob 0.3 --aug-reverb-prob 0.2 \
  --val-shift --max-val 8000 --val-every 1 --skip-final-test \
  --seed 42 --num-workers 4 \
  2>&1 | tee -a "$LOG"
echo "[2.4] train done $(date -Iseconds)" | tee -a "$LOG"
