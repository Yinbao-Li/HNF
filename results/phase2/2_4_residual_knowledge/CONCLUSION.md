# Residual knowledge probe — conclusion

**Claim scale (user-confirmed):** residual has **unnamed** information beyond the physics proposal  
(\(I(Y;R\mid\mathrm{Physics})>0\)); we do **not** require naming the content as a physical mechanism.

**Verdict at that scale: SUPPORTED (exploratory).**  
Stronger “named new physics / exogenous mechanism” scale: **not supported** (see strict sensitivity).

## Evidence for unnamed irreducible information

Frozen `outputs/stead_c2f_imp_RSP/last.pt`. Physics-only peak vs full propose–refine decode:

| Domain | physics-only MAE | +refine MAE | frac windows improved |
|--------|-----------------:|------------:|----------------------:|
| STEAD val P | 0.362 s | 0.095 s | **96.5%** |
| ETHZ dev P | 0.795 s | 0.548 s | **93.2%** |

Under Proposition 1 the refine term cannot create picks outside proposals, so the timing gain is residual information inside the proposal neighborhood. Content need not be labeled.

**Suggested TGRS wording (Method/Discussion only; not Abstract “new physics”):**  
*The bounded refine residual carries information not captured by the physics proposal alone, improving onset timing on STEAD and transferring to ETHZ, without requiring an interpretable physical label for that residual.*

## What failed (stronger scale — do not claim)

Dropping the near-circular `shift_full−prop` feature, linear probes of refine logits do **not** meet the pre-registered bar for named/exogenous “new knowledge” (strict `R²≈0.012` on improvement; no stable incremental \(R^2\)).  
`|ref|` correlates with SNR/distance/P–S gap, but that is descriptive, not a discovery claim. Noise windows can show large `|ref|`.

## Artifacts

- `residual_knowledge_probe.json` — full pre-registered probe (initial feature set)
- `strict_sensitivity.json` — ref-only features (no circular shift)
- `tools/phase2_probe_refine_residual_knowledge.py`

No `main.tex` edits in this commit.
