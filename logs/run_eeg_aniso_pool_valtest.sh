#!/usr/bin/env bash
# Aniso phase_off with pooled val+test: same 31-subject pool as val (selection) and test (report).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
OUT="${OUT:-outputs/eeg/aniso_pool_valtest/phase_off}"
CLIN="${CLIN:-outputs/eeg/aniso_pool_valtest/phase_off_clinical}"
mkdir -p logs "$OUT" "$CLIN"

echo "[pool-retrain] train=57; val=test=val+test pool (~31 subjects) → $OUT"
python -u tools/train_eeg_native.py \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --lr 1e-4 \
  --mmse-weight 0.1 \
  --principle aniso_diffusion \
  --no-rhythm-phase \
  --pool-val-test \
  --no-subject-balanced \
  --arch-tag "eeg_hnf_aniso_phase_off_poolvt" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_pool_valtest_train.log

echo "[pool-retrain] clinical → $CLIN"
python -u tools/run_eeg_clinical_suite.py \
  --checkpoint "$OUT/best.pt" \
  --output-dir "$CLIN" \
  --device "$DEVICE" \
  --no-synthetic \
  2>&1 | tee logs/eeg_aniso_pool_valtest_clinical.log \
  || echo "[warn] clinical failed"

python - <<'PY'
import json
from pathlib import Path
base = Path("outputs/eeg/aniso_pool_valtest")
m = json.loads((base / "phase_off/test_metrics.json").read_text())
old = json.loads(Path("outputs/eeg/aniso_diffusion_ablation/phase_off/test_metrics.json").read_text())
pool = json.loads(Path("outputs/eeg/valtest_pool_eval/VALTEST_POOL_BOARD.json").read_text())
pool_row = next(r for r in pool["rows"] if "phase_off" in r["model"] and "aniso" in r["model"].lower())
comp = {
    "protocol": "pool_val_test: train=57; val selection AND final test = merged val+test (~31)",
    "caveat": "val and test are the SAME subjects — optimistic vs a true holdout",
    "new_pooled": {k: m.get(k) for k in (
        "test_epoch_acc", "test_epoch_auc", "test_subject_accuracy",
        "best_clinical_score", "n_test_subjects", "elapsed_sec")},
    "old_default_test18": {k: old.get(k) for k in (
        "test_epoch_acc", "test_epoch_auc", "test_subject_accuracy",
        "best_clinical_score", "n_test_subjects")},
    "prior_eval_only_pool31_no_retrain": {
        "subject_accuracy": pool_row["subject_accuracy"],
        "ad_ftd_subject_accuracy": pool_row["ad_ftd_subject_accuracy"],
        "subject_auc_macro": pool_row["subject_auc_macro"],
    },
}
(base / "COMPARE.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
print(json.dumps(comp, indent=2))
PY
echo "done → $OUT"
