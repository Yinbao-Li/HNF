# Input filter → frozen c2f pilot

Design: `DESIGN.md`. Train: 3 ep × 40k, filter-only (~20k params), picker frozen.

## Results (STEAD test subset n_ev=2000, n_nz=500, seed=7)

| SNR | filtered P F1 | imp_RSP P F1 | Δ |
|-----|--------------:|-------------:|--:|
| clean | 0.9907 | 0.9909 | -0.0002 |
| 0 dB | 0.7614 | 0.7605 | +0.0009 |
| −5 dB | 0.4696 | 0.4598 | +0.0098 |

## Verdict (pilot)

Δ(−5 dB)=+0.0098. Phase-B bar was ≥+0.10 for success / &lt;+0.05 未修复.
This is a **plumbing + short pilot**, not a full claim run.

No main.tex edits.
