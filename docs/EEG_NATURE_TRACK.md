# EEG Nature-track (locked 2026-08-05)

Same bar as seismic \(\beta_\mathrm{res}\): new physical fact after classical
covariates + independent closure + falsifiable out-of-cohort check.

## 0. Physical probe (this is the paper, not the classifier)

Domain II is the **diffusion probe** in the PNF kernel family (Fig. 1 principle:
wave / diffusion / memory). Electrodes are not the instrument; the instrument is

\[
K(D,\tau_0)\;\propto\;(\det D)^{-1/2}(4\pi\tau_0)^{-3/2}
\exp\!\big(-\Delta x^{\mathsf T}D^{-1}\Delta x\,/\,4\tau_0\big)
\]

on 10–20 geometry. Readouts:

| Probe channel | Symbol | Role (seismic analogue) |
|---------------|--------|-------------------------|
| Spatial Green kernel | \(K(D,\tau_0)\) | Huygens \(K(\gamma,\omega)\) on the path |
| Learned SPD tensor | \(D\) eigs / FA | anisotropic medium, not isotropic mixing |
| Medium density | \(\rho(t)\) | \(\rho(\tau)\) delay-time density |
| Rhythm branches | \(\delta/\theta/\alpha\) stacks | band-tuned probe filters |
| Classical \(\theta/\alpha\) power | Welch | a *different* instrument (voltmeter) |

Frozen aniso `phase_off` probe geometry (checkpoint-global, not per subject):
\(D\) eigs \(\approx(0.0043,\,0.0155,\,0.0625)\), \(\tau_0\approx0.192\),
anisotropy \(D_3/D_1\approx14.5\), mix \(\alpha\approx0.63\).
**Probe-physics ablation:** same backbone, rhythm phase on the Green kernel
*hurts* (0.667 vs 0.833) — the probe is diffusive transport, not oscillatory
interference. \(\delta/\theta/\alpha\) branches already carry band structure.

Residualization is not a clinical trick. It asks: **after subtracting the
voltmeter (\(\theta/\alpha\)+age+sex), does the diffusion probe still read a
medium change?** That leftover \(\rho\) / \(D_\mathrm{eff}\) *is* the probe
claim. Subject-level \(D\) fits are per-subject probe calibration (exploratory;
FTD/AD template split failed).

ds005385 transfer asks whether the **same frozen probe** still returns a stable
person-level medium readout in another cohort (healthy aging), with age moving
the same direction as disease.

## Locked claims (keep)

**Updated after train-fit / held-out mine** (`docs/EEG_PROBE_PUBLISHABLE.md`):

1. **Do not headline leftover vs diagnostic stage on \(N=88\).**  
   Train-fit residualizer → val+test stage contrast **fails** (\(r=-0.09\),
   \(p_\mathrm{perm}=0.63\)). The earlier all-sample \(r\approx 0.23\) mixed discovery
   and confirmation.

2. **Publish: leftover vs cognition beyond the voltmeter (OOS).**  
   v3 probe leftover vs MMSE\|voltmeter on val+test: \(r=-0.37\), \(p=0.038\),
   \(\Delta R^2=+0.10\). Phase-on control \(\Delta R^2\approx 0\).

3. **Publish: two kernels, one residual medium axis** (v3 vs aniso leftover \(r\approx 0.8\)–\(0.93\)).

4. **Frozen transfer → ds005385** remains aging-continuum closure for \(\rho\) ICC
   (aniso 0.75–0.84), not AD diagnosis transfer.

## Negative / exploratory (do not headline)

- Subject-level \(D\) from channel covariance: pre-registered
  **FTD > AD on atrophy-template delta → FAIL** (\(p=0.56\), label inversion).
- \(D_\mathrm{aniso}\) leftover looks strong until winsorized — outlier artifact.
- Literature rank templates ≠ same-subject MRI. Voxelwise meta-maps still open.

## Venue

| Story | Ceiling now |
|-------|-------------|
| Classification 0.833 / \(n_\mathrm{test}=18\) | JNE / TBME |
| v3 leftover + jackknife + ds005385 ρ transfer | **NeuroImage / Brain Communications / A&amp;D DADM** |
| + second AD/FTD EEG site + same-subject MRI | NC / Nature Aging candidate |

## Reproduce

```bash
PYTHONPATH=. python tools/analyze_eeg_structure_residual.py
PYTHONPATH=. python tools/transfer_eeg_rho_ds005385.py --device cuda
PYTHONPATH=. python tools/fit_eeg_subject_diffusion.py
PYTHONPATH=. python tools/plot_eeg_nature_track.py
PYTHONPATH=. python tools/run_eeg_nature_track.py --skip-residual --skip-diffusion --skip-transfer
```

Boards: `outputs/eeg/structure_residual/BOARD.md`,
`outputs/eeg/longitudinal_ds005385_rho/BOARD.md`,
`outputs/eeg/subject_diffusion/BOARD.md`,
`outputs/eeg/NATURE_TRACK_BOARD.md`.
Figure: `docs/figures/eeg/eeg_nature_track.{png,pdf}`.
