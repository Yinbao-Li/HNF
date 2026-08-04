#!/usr/bin/env bash
# Interpretability + discovery pipeline for preferred aniso phase_off checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-outputs/eeg/aniso_diffusion_ablation/phase_off/best.pt}"
DEVICE="${DEVICE:-cuda}"
CLIN="${CLIN:-outputs/eeg/aniso_diffusion_ablation/phase_off_clinical}"
BASE="${BASE:-outputs/eeg/aniso_interpret}"
mkdir -p logs "$BASE" docs/figures/eeg

echo "[aniso-interpret] paper figures"
python -u tools/plot_eeg_aniso_paper_figures.py \
  --checkpoint "$CKPT" \
  --clinical-dir "$CLIN" \
  --fig-dir docs/figures/eeg \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_paper_figures.log

echo "[aniso-interpret] explain panels"
python -u tools/explain_eeg_native.py \
  --checkpoint "$CKPT" \
  --output-dir "$BASE/explain" \
  --split test \
  --num-per-group 2 \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_explain.log
# promote one HC / AD example into docs/figures
cp -f "$BASE"/explain/explain_HC_*.png docs/figures/eeg/aniso_explain_hc.png 2>/dev/null || true
cp -f "$BASE"/explain/explain_AD_*.png docs/figures/eeg/aniso_explain_ad.png 2>/dev/null || true
cp -f "$BASE"/explain/explain_FTD_*.png docs/figures/eeg/aniso_explain_ftd.png 2>/dev/null || true

echo "[aniso-interpret] knowledge cards (train FDR hits)"
python -u tools/build_eeg_knowledge_cards.py \
  --clinical-report "$CLIN/clinical_report.json" \
  --output-dir "$BASE/knowledge_cards" \
  --include-train-only \
  2>&1 | tee logs/eeg_aniso_knowledge_cards.log

echo "[aniso-interpret] temporal-chain modes"
python -u tools/build_eeg_temporal_chain_library.py \
  --checkpoint "$CKPT" \
  --output-dir "$BASE/temporal_chain" \
  --split all \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_temporal_chain.log
cp -f "$BASE/temporal_chain/"*modes*.png docs/figures/eeg/aniso_temporal_chain_modes.png 2>/dev/null || \
  cp -f "$BASE/temporal_chain/"*.png docs/figures/eeg/ 2>/dev/null || true

echo "[aniso-interpret] subject clusters"
python -u tools/run_eeg_subject_cluster.py \
  --checkpoint "$CKPT" \
  --output-dir "$BASE/subject_cluster" \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_subject_cluster.log
cp -f "$BASE/subject_cluster/"*pca*.png docs/figures/eeg/aniso_subject_cluster_pca.png 2>/dev/null || \
  cp -f "$BASE/subject_cluster/"*.png docs/figures/eeg/ 2>/dev/null || true

echo "[aniso-interpret] marker stability (aniso only)"
python -u tools/run_eeg_marker_stability.py \
  --checkpoints "$CKPT" \
  --output-dir "$BASE/marker_stability" \
  --bootstrap 100 \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_marker_stability.log

echo "done → $BASE + docs/figures/eeg/aniso_*"
