# LEMON pre-registration (locked 2026-08-06, before looking at structure correlations)

Frozen ds004504 probes only (`native_v3`, `aniso_phase_off`).  
Voltmeter leftover uses **AHEPA train β** from `outputs/eeg/probe_publishable/PROBE_MINE.json`.  
**Do not refit** residualizers on LEMON.

Primary condition: eyes-closed (`EC`). EO is reliability only.

## Sample

MPI-LEMON healthy young + old with paired preprocessed EEG + MP2RAGE T1 (`n≈202`).  
Age = bin midpoint. Sex: META `1=F, 2=M`.

## Primary structure endpoint (v1)

GMM GM volume / ICV on inv-2 (or skull-stripped UNI).  
**Not** FreeSurfer thickness — that is v2 if H1 is null or weak.

## Hypotheses

| ID | Test | Predicted |
|----|------|-----------|
| **H1** | leftover \(D_\mathrm{eff}\) vs GM/ICV \| age+sex | Spearman \(r<0\) |
| **H2** | leftover \(\rho_\mathrm{std}\) vs GM/ICV \| age+sex | Spearman \(r>0\) |
| **H3** | EC vs EO \(D_\mathrm{eff}\) / \(\rho_\mathrm{std}\) | high Pearson \(r\) (reliability) |
| **H4** | raw \(D_\mathrm{eff}\) vs age | \(r>0\) (aging continuum = disease↑) |
| **H5** | leftover \(D_\mathrm{eff}\) vs TMT-B \| age+sex | \(r>0\); vs CVLT long-delay \(r<0\) |

Controls: leftover vs age ≈ 0; young vs old leftover MW non-significant.

## Failure modes (write them down)

- H3/H4 pass, H1/H2 fail → probe works, tissue proxy too crude → run FastSurfer/SynthSeg.
- H1 passes only in old group → report as exploratory split, not primary.
- Any refit of β on LEMON invalidates leftover transfer.

## Not claimed

AD/FTD diagnosis, AHEPA subtype, Nature-level closure without clinical MRI.
