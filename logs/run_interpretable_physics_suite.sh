#!/usr/bin/env bash
# Best interpretable physics suite on run28 (canonical 800-pt model).
# 1) causal-chain modes  2) reclassify+amplitude  3) magnitude/geo ceiling
set -euo pipefail
cd /media/bob/Work/TRELLIS/HNF
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt}"
OUT="${OUT:-outputs/interpretable_physics_best}"
DEVICE="${DEVICE:-cuda}"
MAX_EVENT="${MAX_EVENT:-1500}"
MAX_CHAIN="${MAX_CHAIN:-800}"

mkdir -p logs "$OUT" \
  "$OUT/causal_chain" \
  "$OUT/reclass" \
  "$OUT/ceiling"

echo "[suite] ckpt=$CKPT  device=$DEVICE  max_event=$MAX_EVENT"
echo "[1/3] causal-chain library"
/home/bob/anaconda3/bin/python -u tools/build_causal_chain_library.py \
  --checkpoint "$CKPT" \
  --split val --max-event "$MAX_CHAIN" --device "$DEVICE" \
  --output-dir "$OUT/causal_chain" \
  2>&1 | tee logs/suite_causal_chain.log

echo "[2/3] reclassify (shape + amplitude + confidence)"
/home/bob/anaconda3/bin/python -u tools/reclassify_causal_physics.py \
  --checkpoint "$CKPT" \
  --split val --max-event "$MAX_EVENT" --k 6 --device "$DEVICE" \
  --output-dir "$OUT/reclass" \
  2>&1 | tee logs/suite_reclass.log

echo "[3/3] interpretable ceiling (mag + geo + taxonomy)"
/home/bob/anaconda3/bin/python -u tools/interpretable_ceiling.py \
  --traces "$OUT/reclass/traces.csv" \
  --output-dir "$OUT/ceiling" \
  2>&1 | tee logs/suite_ceiling.log

# master report
/home/bob/anaconda3/bin/python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
chain = json.loads((out/"causal_chain"/"causal_chain_report.json").read_text()) if (out/"causal_chain"/"causal_chain_report.json").exists() else {}
reclass = json.loads((out/"reclass"/"reclass_report.json").read_text()) if (out/"reclass"/"reclass_report.json").exists() else {}
ceil = json.loads((out/"ceiling"/"ceiling_report.json").read_text()) if (out/"ceiling"/"ceiling_report.json").exists() else {}
h = ceil.get("headline", {})
geo = reclass.get("geography", {})
tax = reclass.get("taxonomies", {})
master = {
  "checkpoint": "$CKPT",
  "canonical_inference": "resample any rate → seq_len=800 → run28 best",
  "two_tier": {
    "router": "pattern_library summary only (no causal chain, no global γ/ω/c)",
    "interpretability": "causal-chain shape + per-trace kernel-row responses (ρ-modulated)",
  },
  "n_reclass": reclass.get("n_traces"),
  "n_ceiling": ceil.get("n_traces"),
  "n_causal_modes": chain.get("k") or len(chain.get("modes", [])),
  "taxonomies": {
    name: {
      "n_modes": len((info or {}).get("modes", [])),
      "names": [m.get("name") for m in (info or {}).get("modes", [])],
    }
    for name, info in tax.items()
  },
  "magnitude": {
    "richter_r2": h.get("richter_r2"),
    "site_r2": h.get("site_r2"),
    "phys_site_r2": h.get("phys_site_r2"),
    "phys_path_site_r2": h.get("phys_path_site_r2"),
    "best_r2": h.get("stratified_site_r2"),
    "best_mae": h.get("stratified_site_mae"),
  },
  "geography": {
    "shape_path_V": h.get("shape_region_V"),
    "shape_path_V_0_50km": h.get("shape_region_V_0_50km"),
    "shape_src_V": h.get("shape_src_region_V"),
    "tax2d_path_V": h.get("tax2d_region_V"),
    "reclass_shape_only_V": (geo.get("shape_only") or {}).get("cramers_v"),
    "reclass_shape_plus_kernel_V": (geo.get("shape_plus_kernel") or {}).get("cramers_v"),
    "reclass_full_interpretable_V": (geo.get("full_interpretable") or {}).get("cramers_v"),
  },
  "taxonomy": "shape×strength (data-driven) + optional shape_plus_kernel k-means",
  "artifacts": {
    "causal_chain": str(out/"causal_chain"),
    "reclass": str(out/"reclass"),
    "ceiling": str(out/"ceiling"),
    "interpretability_md": str(out/"ceiling"/"INTERPRETABILITY.md"),
  },
}
(out/"MASTER_REPORT.json").write_text(json.dumps(master, indent=2))
md = f"""# Interpretable physics (best suite)

Canonical picker: **run28 @ 800** (resample any input rate first).

## Magnitude (single-station, interpretable)
| Model | R² | MAE |
|-------|---:|----:|
| Richter logA+logD | {h.get('richter_r2')} | — |
| + station/network site | {h.get('site_r2')} | — |
| + depth/SNR + site | {h.get('phys_site_r2')} | — |
| + coda path residual + site | {h.get('phys_path_site_r2')} | — |
| **ml/md stratified (best)** | **{h.get('stratified_site_r2')}** | **{h.get('stratified_site_mae')}** |

## Geography / structure
- shape ↔ path-region Cramér's V = **{h.get('shape_region_V')}** (0–50 km: **{h.get('shape_region_V_0_50km')}**)
- shape ↔ source-region V = **{h.get('shape_src_region_V')}**
- tax2d ↔ path-region V = **{h.get('tax2d_region_V')}**
- reclass k-means path V: shape_only **{(geo.get('shape_only') or {}).get('cramers_v')}** /
  shape+kernel **{(geo.get('shape_plus_kernel') or {}).get('cramers_v')}** /
  full **{(geo.get('full_interpretable') or {}).get('cramers_v')}**

## Two-tier
- Router: pattern-library summaries only
- Interpretability: causal-chain shape + per-trace kernel-row responses (not global γ/ω/c)

## Reading
- **strength** (reduced amplitude) → source / magnitude
- **shape** (onset, coda, ρ multipath) → path / mechanism / geography
- Kernel-row features refine mechanism views; they do not replace shape for geo V
- Ceiling note: {h.get('ceiling_note')}

See \`ceiling/INTERPRETABILITY.md\` and plots under \`ceiling/\`, \`reclass/\`, \`causal_chain/\`.
"""
(out/"MASTER_REPORT.md").write_text(md)
print(json.dumps(master, indent=2))
print(f"[suite] wrote {out/'MASTER_REPORT.md'}")
PY

echo "done → $OUT"
