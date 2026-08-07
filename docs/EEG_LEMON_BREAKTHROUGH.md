# LEMON research board — breakthrough mine (2026-08-06)

Locked before looking at structure: `docs/EEG_LEMON_PREREG.md`.  
This board **overrides** any earlier BOARD that used AHEPA-β leftovers on LEMON.

Data: `n=202` EC+EO+T1 (Babayan 2019). Frozen `native_v3` / `aniso_phase_off`.  
Morph: GMM GM/ICV on inv-2 (`outputs/eeg/lemon_morphometry/`) — crude, but GM/ICV declines with age (ρ=−0.39).

---

## Verdict (one paragraph)

LEMON does **not** currently close the probe↔structure story. The important outcome is methodological and negative: **AHEPA train-β leftover cannot be transferred to LEMON in raw units** (scale mismatch → leftover ≈ −voltmeter prediction, dual-probe ρ≈0.99 is an artifact). Under honest within-LEMON residualization, dual-probe agreement stays ~raw (ρ≈0.70), and leftover tracks neither GM/ICV nor TMT/CVLT. Absolute \(D_\mathrm{eff}\) is domain-shifted (LEMON mean 0.35 vs AHEPA 1.42) even with full 19 channels. What survives: EC–EO person-level reliability (v3 \(r\approx0.68\), ICC≈0.66). Age continuum matches disease only on ds005385 / AHEPA elderly HC — not on LEMON’s young–old gap.

---

## KEEP (real findings)

### K1 — Domain shift of the frozen probe (must report)
| cohort | v3 \(D_\mathrm{eff}\) mean±sd | v3 \(\rho_\mathrm{std}\) mean |
|--------|-------------------------------:|------------------------------:|
| AHEPA ds004504 | 1.42 ± 0.71 | 0.92 |
| LEMON EC | **0.35 ± 0.05** | **3.00** |
| LEMON full-19 only | 0.36 ± 0.06 | 2.94 |

Not explained by zero-padding (full-19 still shifted). Likely montage / ICA pipeline / healthy vs clinical amplitude.  
**Claim:** absolute probe readout is not transportable without calibration.

### K2 — AHEPA-β leftover transfer is invalid on LEMON
Applying AHEPA `train_beta` to LEMON:

- `corr(leftover, −pred) = 0.997` (v3)
- leftover SD / raw \(D_\mathrm{eff}\) SD ≈ **12×**
- AHEPA-β dual-probe leftover ρ = **0.993** (CI 0.989–0.994)

This is **not** “two kernels, one medium axis” replicating. It is both leftovers ≈ affine transforms of the same (θ/α, bp_α) under wrong β scale (LEMON bp_α mean 2.14 vs AHEPA 1.06; \(D_\mathrm{eff}\) compressed).

**Kill any figure/BOARD that treated AHEPA-β LEMON leftovers as structure tests.**

### K3 — Honest within-LEMON residualization
LEMON-OLS \(D_\mathrm{eff} \sim\) age+sex+θ/α+bp_α:

| test | v3 | aniso |
|------|---:|------:|
| dual-probe leftover ρ | 0.70 | (same pair) |
| raw dual-probe ρ | 0.68 | — |
| leftover vs GM/ICV \| demo | ~0 | ~0 |
| leftover vs TMT-B \| demo | ~0 | ~0 |
| R²(volt) of \(D_\mathrm{eff}\) | 0.30 | 0.54 |
| ΔR² from adding GM | **0.000** | **0.000** |

Dual-probe “collapse” beyond raw agreement **does not** appear on LEMON when residualized correctly.

### K4 — EC–EO reliability (positive, modest)
v3 raw \(D_\mathrm{eff}\): EC–EO \(r=0.68\), ICC≈0.66 (old ICC≈0.73).  
Person-level trait exists in healthy adults across eye state. Useful as a stability floor, not a Nature claim alone.

### K5 — Age continuum is cohort-conditional
| sample | v3 \(D_\mathrm{eff}\) vs age | vs disease↑? |
|--------|-----------------------------:|:------------:|
| LEMON all (young+old gap) | ρ=−0.19, p=0.006 | **opposite** |
| LEMON young only / old only | ~0 | — |
| after spectral residual on LEMON | ~0 | — |
| ds005385 (n=40, ages 22–70) | r≈+0.14 to +0.26 | matches |
| AHEPA HC only (57–78) | ρ≈+0.23 (ns) | matches |

Young adults drive LEMON reversal; spectral features absorb the age effect. Do **not** claim a universal aging continuum from LEMON.

---

## FAIL (pre-registered)

| ID | Result |
|----|--------|
| H1 leftover vs GM/ICV | null (AHEPA-β invalid; LEMON-OLS also null) |
| H2 leftover vs ρ–GM | null |
| H4 age continuum on LEMON | fail / reverse for v3 |
| H5 leftover vs TMT/CVLT | null (old-only CVLT ρ≈−0.21, p≈0.09 — exploratory only) |

Morph QC: GM/ICV declines with age (good). Absolute `brain_cm3` larger in old (bad) — do not use absolute brain volume.

---

## What this means for Nature / NC

| Path | Status after LEMON |
|------|--------------------|
| Probe leftover ↔ same-subject atrophy (healthy closure) | **Blocked** until calibrated transfer + real thickness (FastSurfer/SynthSeg), or AHEPA MRI table |
| Dual-probe one axis as OOS transfer via β | **Killed** (artifact). Remains AHEPA-internal claim |
| Frozen absolute ρ / \(D_\mathrm{eff}\) as cross-dataset biomarker | **Killed** without domain calibration |
| EC–EO reliability + ds005385 ICC | Supports “person-level readout”, NeuroImage-tier methods |
| AHEPA leftover ↔ MMSE OOS | Still the best clinical claim; LEMON did not strengthen it |

**Breakthrough-grade honest sentence:**

> A frozen AD/FTD diffusion probe is person-stable (EC–EO) but not scale-transportable to an independent 62-channel healthy cohort; naive reuse of clinical residualizers fabricates dual-probe agreement, and calibrated leftovers do not track global GM fraction or cognition in healthy adults — so structural closure, if it exists, must be tested with clinical MRI or thickness, not LEMON GMM + AHEPA-β.

---

## Next experiments (research, not plots)

1. **Domain calibration**: map LEMON (θ/α, bp_α, \(D_\mathrm{eff}\)) onto AHEPA HC marginals (quantile / affine), then re-test leftover vs GM. Pre-register: calibration on demographics only, not on GM.
2. **FastSurfer / SynthSeg** on LEMON T1 → thickness + hippocampus (replace GMM).
3. **AHEPA MRI table** (email) — only path to disease-structure closure.
4. **Diagnose ρ inflation**: compare LEMON vs AHEPA epoch waveforms (RMS, spectra) under identical loader; test whether common-average reference or dropping padded channels fixes scale.
5. Do **not** publish LEMON structure correlations from AHEPA-β leftovers.

Reproduce mining:
```bash
PYTHONPATH=. python tools/run_eeg_lemon_probe.py --skip-forward   # CSV only
# then re-run the scale-check logic in this board / a future tools/mine_eeg_lemon_breakthrough.py
```
