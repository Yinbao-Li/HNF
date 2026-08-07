# Impact discovery lock — EEG diffusion probe (2026-08-06)

Goal: one meaningful claim with the PNF/HNF model, after LEMON falsified naive structure transfer.

## The discovery (publish this)

**A frozen cortical diffusion / Huygens probe reads a person-level “medium” state that is**

1. **orthogonal to the classical EEG voltmeter** (age, sex, θ/α, bp_α; residualizer fit on train only),
2. **shared by two different spatial kernels** (v3 Fresnel vs aniso diffusion leftover Spearman \(r=0.93\) on val+test),
3. **physically diffusion-like**: putting oscillatory phase on the Green kernel (**phase on**) drives held-out MMSE increment to ~0 / negative,
4. **not healthy aging or global GM** (LEMON \(n=202\): within-cohort leftover ⊥ age, ⊥ GM/ICV, ⊥ TMT),
5. **coupled to cognition under neurodegeneration** (AHEPA): train-fit leftover tracks MMSE\|voltmeter on held-out subjects (test Pearson \(r\approx-0.37\), bootstrap 95% CI excludes 0; \(P(r<0)\approx0.98\)).

**One sentence**

> After removing the EEG voltmeter, two frozen PNF spatial probes collapse onto one residual medium axis that is null in healthy lifespan MRI/cognition (LEMON) but tracks MMSE in AD/FTD (AHEPA), and the effect requires a diffusion Green kernel without oscillatory phase.

That is the impactful result: **pathology-coupled medium readout**, not “another AD classifier,” not “correlates with age,” not “GMM thickness.”

---

## What we corrected today (do not oversell)

| Old wording | Correction |
|-------------|------------|
| valtest MMSE \(\Delta R^2=+0.10\) as OOS prediction | That \(\Delta R^2\) is **in-split** OLS \(R^2\) on the 31 held-out people. **Train→held-out predictive** \(\Delta R^2\) is only \(\approx+0.007\). Prefer **correlation** of train-fit leftover with MMSE\|voltmeter (bootstrap CI). |
| AHEPA-β leftover on LEMON dual-probe \(r\approx0.99\) | **Artifact** (scale mismatch; leftover \(\approx-\)voltmeter prediction). Void. |
| LEMON leftover ↔ GM (H1) | **Fail** under honest LEMON-OLS residualization. |
| Universal aging continuum for \(D_\mathrm{eff}\) | LEMON young–old **reverses** v3; ds005385 / AHEPA elderly HC match disease direction. Cohort-conditional only. |

---

## Evidence table (locked)

### Physics (P1)
| contrast | held-out MMSE increment |
|----------|-------------------------|
| aniso **phase_off** | HO nested \(\Delta R^2\) small positive / association present |
| aniso **phase_on** | HO \(\Delta R^2\) **negative** (bootstrap mean \(-0.046\), \(P(\Delta>0)\approx0.09\)) |

### Dual probe (P2)
val+test leftover–leftover \(r=0.93\); second probe adds little unique MMSE.

### Cognition (P3, honest)
| metric | train | val+test | test |
|--------|------:|---------:|-----:|
| predictive \(\Delta R^2\) (train β → HO) | +0.138 | **+0.007** | +0.033 |
| Pearson \(r\)(leftover, MMSE\|volt\(_\mathrm{train}\)) | −0.41 | **−0.30** (boot CI excludes 0) | −0.37 (boot CI excludes 0) |

### Healthy specificity (NEW, LEMON)
LEMON-OLS leftover vs age / GM/ICV / TMT ≈ 0.  
Absolute probe scale is domain-shifted (LEMON \(D_\mathrm{eff}\) mean 0.35 vs AHEPA 1.42) — report as methods constraint, not a disease claim.

---

## Why this can still matter (venue)

| Venue | Fit |
|-------|-----|
| **NeuroImage / Network Neuroscience / Brain Comm** | Yes: physics ablation + dual-probe + healthy-null specificity + held-out cognition association |
| **Nature / NC** | **Not yet** — need same-subject clinical MRI (AHEPA table) or second clinical EEG site with calibrated transfer |
| Classifier papers on ds004504 | We are not competing on accuracy; we are claiming a **different instrument** |

Nature bar remains: calibrated leftover ↔ hippocampal / thickness **in patients**, Steiger vs θ/α, phase ablation retained.

---

## What to do next (only high EV)

1. **Send AHEPA MRI-table email** — only path to NC-tier closure.  
2. Optional: FastSurfer on LEMON for thickness (expect still weak if leftover is pathology-coupled).  
3. Optional: domain-calibrated ranks LEMON↔AHEPA HC (already tried quantile map → still null vs GM).  
4. **Do not** chase LEMON structure with AHEPA-β leftovers.  
5. Rewrite paper spine around **pathology-coupled medium**, with LEMON as negative control.

Reproduce:
```bash
PYTHONPATH=. python tools/mine_eeg_probe_publishable.py
# LEMON honesty: docs/EEG_LEMON_BREAKTHROUGH.md
```
