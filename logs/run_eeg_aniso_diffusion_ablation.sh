#!/usr/bin/env bash
# Anisotropic diffusion kernel for EEG + rhythm-phase ablation (on vs off).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
MMSE_W="${MMSE_W:-0.1}"
BASE="${BASE:-outputs/eeg/aniso_diffusion_ablation}"

mkdir -p logs "$BASE"

run_one () {
  local tag="$1"
  local phase_flag="$2"
  local out="${BASE}/${tag}"
  local clin="${BASE}/${tag}_clinical"
  mkdir -p "$out" "$clin"
  echo "[aniso-eeg] === ${tag} → ${out}  ${phase_flag} ==="
  python -u tools/train_eeg_native.py \
    --output-dir "$out" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --mmse-weight "$MMSE_W" \
    --principle aniso_diffusion \
    ${phase_flag} \
    --no-subject-balanced \
    --arch-tag "eeg_hnf_aniso_${tag}" \
    --no-synthetic \
    2>&1 | tee "logs/eeg_aniso_${tag}_train.log"

  echo "[aniso-eeg] clinical suite → ${clin}"
  python -u tools/run_eeg_clinical_suite.py \
    --checkpoint "${out}/best.pt" \
    --output-dir "$clin" \
    --device "$DEVICE" \
    --no-synthetic \
    2>&1 | tee "logs/eeg_aniso_${tag}_clinical.log" \
    || echo "[aniso-eeg] WARN clinical suite failed for ${tag} (training metrics kept)"
}

# Sequential on one GPU: phase ON then phase OFF.
run_one "phase_on" "--rhythm-phase"
run_one "phase_off" "--no-rhythm-phase"

python - <<'PY'
import json
from pathlib import Path
base = Path("outputs/eeg/aniso_diffusion_ablation")
rows = []
for tag in ("phase_on", "phase_off"):
    p = base / tag / "test_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        rows.append({"tag": tag, **{k: m.get(k) for k in (
            "test_epoch_acc", "test_epoch_auc", "test_subject_accuracy",
            "best_clinical_score", "n_params", "elapsed_sec")}})
board = {"ablation": "aniso_diffusion rhythm_phase", "runs": rows}
(base / "ABLATION_BOARD.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
md = ["# Anisotropic diffusion × rhythm-phase ablation", ""]
md.append("| tag | epoch_acc | epoch_auc | subject_acc | clinical_score | params |")
md.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    md.append(
        f"| {r['tag']} | {r.get('test_epoch_acc', float('nan')):.4f} | "
        f"{r.get('test_epoch_auc', float('nan')):.4f} | "
        f"{r.get('test_subject_accuracy', float('nan')):.4f} | "
        f"{r.get('best_clinical_score', float('nan')):.4f} | "
        f"{r.get('n_params', '')} |"
    )
(base / "ABLATION_BOARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps(board, indent=2))
PY

echo "done → ${BASE}/ABLATION_BOARD.md"
