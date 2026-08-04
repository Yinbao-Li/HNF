#!/usr/bin/env bash
# Retrain aniso diffusion (phase_off) with original test used as val (swap_val_test).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
OUT="${OUT:-outputs/eeg/aniso_swap_valtest/phase_off}"
CLIN="${CLIN:-outputs/eeg/aniso_swap_valtest/phase_off_clinical}"
mkdir -p logs "$OUT" "$CLIN"

echo "[swap-retrain] train→$OUT  (val=old test n=18, test=old val n=13)"
python -u tools/train_eeg_native.py \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --lr 1e-4 \
  --mmse-weight 0.1 \
  --principle aniso_diffusion \
  --no-rhythm-phase \
  --swap-val-test \
  --no-subject-balanced \
  --arch-tag "eeg_hnf_aniso_phase_off_swapvt" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_swap_valtest_train.log

echo "[swap-retrain] clinical → $CLIN"
python -u tools/run_eeg_clinical_suite.py \
  --checkpoint "$OUT/best.pt" \
  --output-dir "$CLIN" \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_swap_valtest_clinical.log

python - <<'PY'
import json
from pathlib import Path
base = Path("outputs/eeg/aniso_swap_valtest")
m = json.loads((base / "phase_off/test_metrics.json").read_text())
old = json.loads(Path("outputs/eeg/aniso_diffusion_ablation/phase_off/test_metrics.json").read_text())
comp = {
    "protocol": "swap_val_test: train fixed; val=original test(18); test=original val(13)",
    "new": {k: m.get(k) for k in (
        "test_epoch_acc", "test_epoch_auc", "test_subject_accuracy",
        "best_clinical_score", "n_test_subjects", "elapsed_sec")},
    "old_default_split_test18": {k: old.get(k) for k in (
        "test_epoch_acc", "test_epoch_auc", "test_subject_accuracy",
        "best_clinical_score", "n_test_subjects", "elapsed_sec")},
}
(base / "COMPARE_TO_DEFAULT.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
md = [
    "# Aniso phase_off retrain with test→val swap",
    "",
    "- Train: same 57 subjects",
    "- Val (selection): **original test** (18)",
    "- Test (report): **original val** (13)",
    "",
    "| setting | subject acc | epoch auc | epoch acc | clinical_score | n_test |",
    "|---|---:|---:|---:|---:|---:|",
    f"| default (test=18) | {old.get('test_subject_accuracy', float('nan')):.3f} | "
    f"{old.get('test_epoch_auc', float('nan')):.3f} | {old.get('test_epoch_acc', float('nan')):.3f} | "
    f"{old.get('best_clinical_score', float('nan')):.3f} | {old.get('n_test_subjects', 18)} |",
    f"| **swap (test=old val 13)** | {m.get('test_subject_accuracy', float('nan')):.3f} | "
    f"{m.get('test_epoch_auc', float('nan')):.3f} | {m.get('test_epoch_acc', float('nan')):.3f} | "
    f"{m.get('best_clinical_score', float('nan')):.3f} | {m.get('n_test_subjects', 13)} |",
    "",
]
(base / "COMPARE_TO_DEFAULT.md").write_text("\n".join(md), encoding="utf-8")
print(json.dumps(comp, indent=2))
PY
echo "done → $OUT"
