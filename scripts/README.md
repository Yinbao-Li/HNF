# HNF scripts layout

Run all commands from the **repo root** (`HNF/`), so `outputs/`, `STEAD/`, and
`hnf/` resolve correctly:

```bash
cd /path/to/HNF
python scripts/paper/run_paper_fig1_overview.py
```

| Directory | Contents |
|-----------|----------|
| `experiments/` | Numbered STEAD picking launches **run11–run27** (archive / ablation history). Primary train entry is still root `run28_stead_ms_fresnel_phys.py`. |
| `paper/` | Paper figure / board drivers (`run_paper_*`). |
| `inversion/` | Classic 1D inv, Zhizi eval helpers, Phase E/F / perf sweeps. |
| `picking/` | Diagnostics (wrong-peak, realtime, Fresnel iterate, …). |
| `domain/` | Non-seismic helpers (e.g. EEG analysis). |

Day-to-day entry points stay at repo root: `train_*.py`, `eval_*.py`,
`run28_*`, `run_interpret_suite.py`, `run_probing_suite.py`,
`run_route_a2_waveform.py`, `run_proof_suite.py`, `run_knowledge_mining*.py`,
`download_*.py`.
