# Coda waveform facies map path-attenuation domains that velocity and direct-wave \(Q\) do not resolve

**Target:** *Nature Communications*  
**Status:** research draft (numbers frozen 2026-08-05; jackknife + tomography gates closed)  
**Do not re-threshold the 5-class taxonomy after seeing maps.**

---

## Abstract

High-frequency attenuation is usually reduced to a single quality factor. We show that an interpretable taxonomy of local-earthquake coda facies retains spatially coherent path information after distance, depth, magnitude, mapped-fault proximity and station effects are removed. In southern California, path midpoints in the Salton–Imperial junction decay faster than the *same stations* record elsewhere (20 stations, site-median \(\Delta\beta \approx -0.032\)), while midpoints in the eastern Transverse Ranges ring longer (\(\Delta\beta \approx +0.025\)). The two domains occupy the same Berg et al. (2021) upper-crust \(V_S\) band yet opposite \(\beta_{\mathrm{res}}\), and they are orthogonal to—even anti-aligned with—Lin & Jordan (2023) direct-wave \(Q_S\). A classical single-trace coda-decay slope explains only about half of each anomaly, and multipath versus slow-coda facies still separate inside matched \(Q_c\) terciles. The same pipeline recovers a Salton-signed faster-decay residual at Mt St Helens–Cowlitz (8 stations; source- and year-jackknife stable) and an opposite, slower-decay residual in the Seattle basin. Direct-wave \(Q\), \(V_S\) and coda facies therefore unmix different path physics in the same crust.

## Introduction

Maps of crustal \(V_S\) and of direct-wave attenuation \(Q_P,Q_S\) are mature products in southern California[^berg2021][^lin2023][^lee2014]. Coda amplitudes, by contrast, are still most often collapsed to a scalar coda quality factor \(Q_c\) or to a site term. That compression is convenient, but it mixes absorption, scattering, multi-pathing and near-receiver resonance—physics that need not share a map.

Lin & Jordan (2023) already emphasise that high-frequency *direct-wave* \(Q_S\) disagrees with coda-derived attenuation and is better explained by elastic scattering than by anelastic loss[^lin2023]. If that is right, coda envelopes should carry a spatially organised residual that velocity models and direct-wave \(Q\) do not absorb. Testing that statement requires (i) a frozen, interpretable description of coda shape, not a post-hoc cluster label, (ii) a structure expectation that removes the obvious geometric and station effects, and (iii) independent published cubes against which to ask whether any leftover map merely retraces \(V_S\) or \(Q_S\) provinces.

Here we assign each STEAD local-earthquake waveform[^mousavi2019] to one of five facies using ceilings on observables from a frozen physics-constrained picker (impulsive fast-\(Q\), emergent, multipath, slow coda, standard). We then form a distance-detrended coda-slope residual \(\beta\) and remove a ridge expectation on \(\log_{10}\) distance, depth, magnitude, log-distance to the nearest Quaternary fault, and station. The leftover field \(\beta_{\mathrm{res}}\) is evaluated at path midpoints. We ask three falsifiable questions:

1. Do opposite-sign \(\beta_{\mathrm{res}}\) domains survive a same-station test?
2. Do they collapse after regression on classical \(Q_c\), Berg \(V_S\), or Lin \(Q_S\)?
3. Does a Salton-signed residual appear outside southern California under the same pipeline?

## Results

### A frozen five-class coda taxonomy

Facies are assigned with ceilings fixed before any regional map was inspected (peak threshold, coda-slope bounds, onset sharpness; Supplementary Table 1). They are not \(k\)-means labels. Multipath traces show multi-lobe delay-time density \(\rho(\tau)\); slow-coda traces keep energy later after \(S\); impulsive-fast\(Q\) traces combine sharp onsets with steep coda decay (Fig. 1). STEAD spans 1984–2018 and is dominated by shallow crustal events (median depth \(\approx 8\) km).

### Structure residuals in southern California

On a fixed-station SoCal panel (\(n=8294\) traces, 20–120 km), the structure model explains \(R^2=0.27\) of \(\beta\). Most of the interpretable attenuation fingerprint is therefore *not* distance, depth, magnitude, fault proximity or station. Gridding \(\beta_{\mathrm{res}}\) at \(0.25^\circ\) path-midpoint cells yields two opposite-sign anomalies that pass a same-station test (Fig. 2; Supplementary Table 2):

- **Salton / southern San Andreas–Imperial junction** (−116.38°, 33.38°): 20 stations, site-median \(\Delta\beta = -0.032\), 80% of stations agree in sign, Cohen’s \(d=-0.54\), Mann–Whitney \(p=6.6\times10^{-32}\). Faster coda decay than the same stations record on other paths.
- **Eastern Transverse Ranges** (−116.38°, 33.88°): 15 stations, \(\Delta\beta = +0.025\), \(d=+0.68\), \(p=2.5\times10^{-12}\). Longer ringing.

Same-station envelope stacks confirm the sign: Salton in-cell envelopes decay earlier; Transverse-Range in-cell energy persists later (Extended Data Fig. 1). These are path-structure residuals, not seismicity-rate patches.

### More than a single \(Q_c\), and not a \(V_S\) province

A classical coda-decay slope \(q_c\) (log-energy versus time in an 8 s window after an energy-based \(S\) proxy) correlates with \(\beta_{\mathrm{res}}\) at \(r=0.64\) (\(n=5000\)). Residualizing \(\beta_{\mathrm{res}}\) on \(q_c\) halves but does not remove the cell anomalies (Salton \(\Delta\beta\): −0.030 → −0.014; Transverse Ranges: +0.040 → +0.019). Inside every \(q_c\) tercile, slow-coda traces still have higher median \(\beta_{\mathrm{res}}\) than multipath traces (Fig. 3a,b). The five-class taxonomy is therefore not a re-wrapping of scalar \(Q_c\).

Sampling Berg et al. (2021) upper-crust \(V_S\) at path midpoints (\(n=6605\) inside the model) gives \(r(\beta_{\mathrm{res}}, V_S^{0-8\,\mathrm{km}})=-0.08\). Vertical \(V_S\) profiles at the two cell centres are nearly coincident below \(\sim 3\) km. In the shared band \(V_S\in[3.05,3.21]\) km s\(^{-1}\), Salton still has \(\beta_{\mathrm{res}}=-0.011\) (\(n=194\)) versus \(+0.035\) in the Transverse Ranges (\(n=126\); \(\Delta=0.046\)). Removing \(V_S\) leaves cell \(\Delta\beta\) essentially unchanged (Fig. 3c). Near-surface \(V_P/V_S\) differs more (1.58 vs 1.88) but overlapping tails still split by \(\Delta\beta=0.040\), and residualizing on \(V_P/V_S\) leaves −0.025 / +0.031.

### Orthogonal to direct-wave \(Q_S\)

Lin & Jordan (2023) 3-D \(Q_S\), sampled only on resolved nodes, is likewise nearly orthogonal to \(\beta_{\mathrm{res}}\) (\(r=-0.07\), \(n=8083\)). The two cells occupy non-overlapping \(Q_S^{0-8\,\mathrm{km}}\) bands (medians 794 vs 590). Removing \(Q_S\) leaves \(\Delta\beta\) intact (−0.030 → −0.026; +0.038 → +0.038). The sign is anti-naive: the Salton cell’s nearest resolved node has \(Q_S\approx 1000\) at 5 km (weaker direct-wave attenuation than the 1-D average) but faster coda decay. This is the map-space expression of Lin & Jordan’s statement that direct-wave \(Q_S\) and coda attenuation disagree (Fig. 3d).

### Cascadia replica and an opposite-sign basin

Repeating the pipeline on a dense Cascadia volcanic-arc panel (\(n=9393\) traces, 113 stations, \(R^2=0.28\)) recovers a Salton-signed residual at **Mt St Helens–Cowlitz** (−122.38°, 46.38°): 8 stations, site-median \(\Delta\beta=-0.014\), 88% sign agreement, \(d=-0.30\), \(p=5.6\times10^{-14}\) (Fig. 4). The cell contains 911 traces and is source-clustered (top source bin 13%, top five 36%). Jackknives remain negative and significant after dropping the top 1–10 source bins, capping each bin at 10–40 traces, leaving each year out, and restricting in-cell events to \(M\ge 1.0\) or \(1.5\) (where \(\Delta\beta\) strengthens to −0.035 / −0.039). Verdict: **stable**, not a swarm artefact.

A second Cascadia cell, **Seattle basin** (−122.38°, 47.38°), has the *opposite* sign: 9 stations, \(\Delta\beta=+0.045\), \(d=+0.83\). Two sedimentary/volcanic basins therefore do not share a coda facies. Scalar “basin = low \(Q\)” language cannot host both results.

## Discussion

The simplest reading is that \(\beta_{\mathrm{res}}\) is just another \(Q\) map. Three independent comparisons reject that reading. Classical single-trace \(q_c\) shares about half the geographic signal but leaves a within-band facies split. Published upper-crust \(V_S\) is split, not retraced. Published direct-wave \(Q_S\) is orthogonal and locally anti-aligned—consistent with direct phases sensing total extinction along the geometric ray, while coda facies sense how energy is trapped, scattered and released after \(S\).

We therefore interpret the Salton / St Helens residuals as *absorbing or rapidly de-trapping* paths, and the eastern Transverse Ranges / Seattle residuals as *ringing* paths, relative to the same stations’ other paths. The physical carriers (fluids, damage, basin reverberation, volcanic scatterers) are not uniquely identified by STEAD alone; the claim is the existence of a station-controlled, tomography-irreducible coda-facies field and its out-of-region replication.

A falsifiable prediction follows. Any new SCSN path whose midpoint falls in the Salton cell (−116.38±0.125°, 33.38±0.125°) should show \(\beta_{\mathrm{res}}\) lower than that station’s contemporaneous non-cell paths, with site-median \(\Delta\beta \le -0.02\) once \(n_{\mathrm{stations}}\ge 8\). The Seattle cell predicts the opposite inequality.

Limitations: STEAD is a curated global compilation, not a complete regional catalogue; depth sampling is shallow-crustal; Cascadia has no Lin/Berg cube in this study; Alaska deep-slab paths are a stress test, not a replica. Composite multi-facies “anomaly scores” inflate where a class is nearly absent and are not used for claims.

## Methods (summary)

**Picker and facies.** Run-28 Huygens/Fresnel picker, checkpoint frozen. Causal-chain observables yield coda slope, onset sharpness and \(\rho(\tau)\) peak count. Ceiling thresholds: peak 1.0, coda-fast −0.198, coda-slow −0.113, onset-high 0.877, onset-low 0.733. \(\beta\) is the residual of coda slope on \(\log_{10}(\mathrm{distance}+1)\).

**Structure expectation.** Ridge regression (\(\lambda=8\)) of \(\beta\) on \(\log_{10}\) distance, depth, magnitude, \(\log(1+\mathrm{fault\ distance})\), and station fixed effects. Fault distance uses USGS QFaults polylines, densified and converted with a local km scaling. Path midpoint is the geographic mean of source and receiver.

**Same-station test.** In-cell traces versus traces at the same station whose midpoints lie outside the cell and a 0.5-cell buffer. Report site-median \(\Delta\beta\), sign-agreement fraction, Cohen’s \(d\) and Mann–Whitney \(p\) on pooled paired traces. Discovery threshold: \(\ge 8\) stations and \(p<0.05\).

**Classical \(q_c\).** Mean-square envelope, 21-sample smoother; slope of \(\log_{10} E(t)\) from the first post-peak drop below 55% of peak energy through +8 s.

**Berg \(V_S\).** IRIS EMC `SoCal-BergEtAl2021-UpperCrustVsandVpVs.r0.0.nc`. Trilinear interpolation in depth–lat–lon; column mean 0–8 km at path midpoints.

**Lin \(Q_S/Q_P\).** SCAT text grids[^scatrepo]. Per-depth linear interpolation on resolved nodes only (`resolution=1`); 0–8 km column mean. Unresolved nodes are not infilled with the 1-D average.

**Cascadia expansion.** Same frozen ceilings; STEAD box 45.2–48.8°N, 124.6–120.2°W; min. 2 station-years / 8 traces; 16k labelled traces, 9393 after distance/fixed-station filters.

**Jackknife.** St Helens cell source bins at 0.05°; drop top \(k=1,3,5,10\) bins; cap 40/20/10 traces per bin; leave-one-year-out 2005–2018; in-cell \(M\ge 0.5,1.0,1.5\).

## Data availability

STEAD waveforms as released by Mousavi et al. USGS QFaults GeoJSON. Berg 2021 model via IRIS EMC. Lin & Jordan 2023 SCAT grids: https://github.com/yupinlin/SouthernCaliforniaQmodel. Analysis tables and figures: `outputs/structure_residual_socal/`, `outputs/structure_residual_cascadia_volc/`.

## Code availability

HNF repository tools: `analyze_structure_residual_anomalies.py`, `validate_structure_anomaly_cells.py`, `compare_beta_vs_classical_qc.py`, `overlay_berg2021_vs.py`, `overlay_lin2023_q.py`, `jackknife_sthelens_cell.py`, `expand_shape_labels_regional.py`.

## Figure plan

- **Fig. 1** Existing HNF taxonomy figure (pie, \(\rho(\tau)\), waveforms). Caption: frozen ceilings, not clustering.
- **Fig. 2** SoCal \(\beta_{\mathrm{res}}\) map + same-station forest for #2/#5.
- **Fig. 3** Four-way independence: classical \(q_c\); within-\(q_c\) shape split; Berg \(V_S\) split; Lin \(Q_S\) orthogonal/anti-aligned.
- **Fig. 4** Cascadia map; St Helens and Seattle same-station bars; jackknife strip.

Extended Data: envelope gallery; Alaska stress-test map; threshold freeze log; full jackknife table.

## Prediction (pre-registered)

Salton cell (−116.375±0.125°, 33.375±0.125°): new SCSN paths, same-station \(\Delta\beta \le -0.02\) at \(\ge 8\) stations.  
Seattle cell (−122.375±0.125°, 47.375±0.125°): \(\Delta\beta \ge +0.02\) under the same rule.

[^mousavi2019]: Mousavi et al., *IEEE TGRS* / STEAD release, 2019.
[^berg2021]: Berg et al., *Geophys. Res. Lett.* (2021). doi:10.1029/2021GL092626. IRIS EMC doi:10.17611/dp/emc.2021.scabergetal.1.
[^lin2023]: Lin & Jordan, *Earth Planet. Sci. Lett.* **616**, 118227 (2023).
[^lee2014]: Lee et al., CVM-S4.26, 2014.
[^scatrepo]: https://github.com/yupinlin/SouthernCaliforniaQmodel
