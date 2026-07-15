# HNF scripts layout

Run all commands from the **repo root** (`HNF/`), so `outputs/`, `STEAD/`, and
`hnf/` resolve correctly:

```bash
cd /path/to/HNF
python scripts/interpret/run_probing_suite.py --device cuda
```

| Directory | Contents |
|-----------|----------|
| `experiments/` | Numbered STEAD picking launches **run11–run28** (primary: `run28_stead_ms_fresnel_phys.py`) |
| `interpret/` | `run_interpret_suite`, `run_probing_suite`, `run_knowledge_mining*` |
| `inversion/` | Classic 1D inv, Zhizi helpers, `run_proof_suite`, `run_route_a*`, Phase E/F / perf |
| `paper/` | Paper figure / board drivers (`run_paper_*`) |
| `picking/` | Diagnostics (wrong-peak, realtime, Fresnel iterate, …) |
| `domain/` | Non-seismic helpers (e.g. EEG analysis) |

Repo root keeps only `train_*` / `eval_*` / `download_*` / a few picking helpers
(`analyze_` / `explain_` / `ablation_` / `benchmark_`).
