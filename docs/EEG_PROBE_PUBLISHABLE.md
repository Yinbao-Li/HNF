# EEG probe — what is actually publishable (2026-08-05 mine)

Instrument = PNF physical probe (Huygens v3 / anisotropic diffusion), not the
3-class head. Voltmeter = age + sex + classical \(\theta/\alpha\) + bp\(_\alpha\).
Residualizer **fit on train only**, then applied to held-out subjects.

## Keep (paper spine)

### P1 — Probe physics control
Same aniso backbone: Green-kernel **phase off** vs **phase on**.  
Phase on kills leftover–label, leftover–MMSE, and MMSE \(\Delta R^2\) (\(\approx 0\)).
The readout is diffusive transport, not kernel oscillation.

### P2 — Two kernels, one residual medium axis
Train-fit leftovers: v3 vs aniso Spearman \(r=0.74\) (train) / **\(0.93\)** (val+test).
Unique-stage signal after partialling the other probe is null.  
Second probe adds \(\Delta R^2 \le 0.01\)–\(0.04\) to MMSE beyond the first.  
**Claim:** Fresnel-spatial and diffusion-spatial probes read the *same*
voltmeter-orthogonal medium axis; v3 is the cleaner cognitive readout.

### P3 — Held-out cognition beyond the voltmeter (main result)
v3 probe leftover vs MMSE after removing the voltmeter.

**Honesty update (2026-08-06):** the published-looking \(\Delta R^2=+0.10\) on val+test is
**in-split** OLS \(R^2\) on those 31 subjects (leftover still train-fit).  
**Train→held-out predictive** \(\Delta R^2\) is only \(\approx +0.007\). Prefer correlation:

| split | \(n\) | in-split \(\Delta R^2\) | predictive \(\Delta R^2\) | \(r\)(leftover, MMSE\|volt\(_\mathrm{train}\)) |
|-------|------:|------------------------:|--------------------------:|-----------------------------------------------:|
| train | 57 | +0.138 | +0.138 | Pearson \(-0.41\) |
| **val+test** | **31** | **+0.102** | **+0.007** | Pearson \(-0.30\) (boot CI excludes 0) |
| test only | 18 | +0.136 | +0.033 | Pearson \(-0.37\) (boot CI excludes 0) |

Raw leftover vs MMSE on valtest is near zero (HC MMSE censored at 30);
the association appears **after subtracting spectral slowing**.  
LEMON healthy-null (leftover ⊥ age/GM/TMT) supports pathology-coupling, not generic atrophy
(`docs/EEG_IMPACT_DISCOVERY.md`).

### P4 — Boundary predictions (not subtypes)
Train-fit leftover flags (examples):
- `sub-039` (test HC, MMSE 30, leftover \(+0.83\)) — disease-like probe
- `sub-027` (test AD, MMSE 16, leftover \(-0.71\)) — HC-like probe  

Falsifiable with follow-up / amyloid / MRI. Do not call them discovered subtypes.

## Kill / downgrade

| Previous wording | Status |
|------------------|--------|
| All-\(N=88\) leftover vs stage \(r\approx 0.23\) | **Discovery contamination.** Train-fit → valtest stage \(r=-0.09\), \(p_\mathrm{perm}=0.63\). **Do not claim OOS diagnostic leftover.** |
| Aniso leftover vs stage | Train already \(p_\mathrm{perm}=0.10\); valtest null |
| Subject-\(D\) FTD/AD atrophy-template split | Pre-registered **FAIL** |
| Classification 0.833 | Methods supplement only |

## One-sentence claim

> A frozen PNF diffusion/Huygens probe of cortical medium-density dynamics
> explains cognitive (MMSE) variance beyond classical EEG slowing and
> demographics in held-out subjects; two probe kernels collapse to one residual
> axis; kernel phase ablation abolishes the effect; the same leftover does
> **not** confirm diagnostic labels out of sample.

## Venue

**NeuroImage / Brain Communications / Network Neuroscience** — physical-probe +
cognition increment + OOS + ablation.  
Not NC on this cohort (no second AD site, no imaging, diagnosis OOS failed).

## Reproduce

```bash
PYTHONPATH=. python tools/mine_eeg_probe_publishable.py
PYTHONPATH=. python tools/plot_eeg_probe_publishable.py
```

Board: `outputs/eeg/probe_publishable/BOARD.md`  
Figure: `docs/figures/eeg/eeg_probe_publishable.{png,pdf}`
