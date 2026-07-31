#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
OUT_ROOT=outputs/rheo/suite_final
mkdir -p "$OUT_ROOT" logs
LOG=logs/rheo_memory_suite.log
echo "[suite] resume $(date -Is)" | tee -a "$LOG"

run() {
  local name="$1"; shift
  if [[ -f "$OUT_ROOT/$name/summary.json" ]]; then
    echo "[skip] $name already done" | tee -a "$LOG"
    return 0
  fi
  echo "===== $name =====" | tee -a "$LOG"
  python -u tools/train_rheo_memory.py --tag "$name" --output-dir "$OUT_ROOT/$name" "$@" >>"$LOG" 2>&1
  python -c "import json; s=json.load(open('$OUT_ROOT/$name/summary.json')); t=s['test']; print('[%s] stress=%.4f lam=%.4f G=%.4f score=%.4f' % ('$name', t['stress_rel'], t['lambda_rel'], t['G_rel'], t['score']))" | tee -a "$LOG"
}

COMMON=(--epochs 45 --batch-size 32 --n-train 1280 --n-val 160 --n-test 160
        --n-steps 160 --dt 0.05 --lr 5e-3 --noise-std 0.01 --device cpu)

run r0_k1_stress_only "${COMMON[@]}" --n-modes 1 --dim 1 --freq-weight 0 --param-weight 0 --param-reg 0
run r0_k2_stress_only "${COMMON[@]}" --n-modes 2 --dim 1 --freq-weight 0 --param-weight 0 --param-reg 0
run r0_k2_freq "${COMMON[@]}" --n-modes 2 --dim 1 --freq-weight 0.2 --param-weight 0 --param-reg 0.01
run r0_k2_full "${COMMON[@]}" --n-modes 2 --dim 1 --freq-weight 0.2 --param-weight 0.5 --param-reg 0.01
run r0_k3_full "${COMMON[@]}" --n-modes 3 --dim 1 --freq-weight 0.2 --param-weight 0.5 --param-reg 0.01
run r1_k2_aniso "${COMMON[@]}" --n-modes 2 --dim 2 --anisotropic --freq-weight 0 --param-weight 0.5 --param-reg 0.01

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/rheo/suite_final")
rows = []
for p in sorted(root.glob("*/summary.json")):
    s = json.loads(p.read_text())
    t = s.get("test", {})
    rows.append({
        "name": s.get("tag", p.parent.name),
        "stress_rel": t.get("stress_rel"),
        "lambda_rel": t.get("lambda_rel"),
        "G_rel": t.get("G_rel"),
        "score": t.get("score"),
        "params": s.get("params"),
        "material_gt": s.get("material_gt"),
        "best_val": s.get("best_val"),
    })
rows.sort(key=lambda r: r["score"] if r["score"] is not None else 1e9)
board = {"root": str(root), "rows": rows, "best": rows[0] if rows else None}
(root / "BOARD.json").write_text(json.dumps(board, indent=2))
lines = [
    "# Rheo Boltzmann memory suite",
    "",
    "| Model | stress_rel | λ_rel | G_rel | score |",
    "|-------|----------:|------:|------:|------:|",
]
for r in rows:
    lines.append(
        f"| {r['name']} | {r['stress_rel']:.4f} | {r['lambda_rel']:.4f} | {r['G_rel']:.4f} | {r['score']:.4f} |"
    )
b = rows[0]
lines += [
    "",
    f"**Best:** `{b['name']}` (score={b['score']:.4f})",
    f"- learned: `{json.dumps(b['params'])}`",
    f"- GT: `{json.dumps(b['material_gt'])}`",
]
(root / "BOARD.md").write_text("\n".join(lines))
print("\n".join(lines))
PY
echo DONE | tee -a "$LOG"
