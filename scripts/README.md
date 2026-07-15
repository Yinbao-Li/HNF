# HNF scripts layout

Run all commands from the **repo root** (`HNF/`), so `outputs/`, `STEAD/`, and
`hnf/` resolve correctly:

```bash
cd /path/to/HNF
python scripts/interpret/run_probing_suite.py --device cuda
python tools/eval_stead_picking.py --checkpoint outputs/run28/.../best.pt
```

| Directory | Contents |
|-----------|----------|
| `experiments/` | Numbered STEAD picking launches **run11–run28** |
| `interpret/` | interpret / probing / knowledge mining suites |
| `inversion/` | Classic 1D inv, proof, route A/A2, Phase E/F / perf |
| `paper/` | Paper figure / board drivers (`run_paper_*`) |
| `picking/` | Diagnostics (wrong-peak, realtime, Fresnel iterate, …) |
| `domain/` | Non-seismic helpers (e.g. EEG analysis) |

Train / eval / download CLIs live in [`../tools/`](../tools/).
