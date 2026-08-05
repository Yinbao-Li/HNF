# Nature-family submission brief — interpretable seismic facies residuals

**Working title**  
Waveform facies unmix path-attenuation domains that a single coda-Q parameter cannot resolve

**Target (honest)**  
- *Nature Geoscience* / *Nature Communications* **only if** Gate A–C below all close.  
- Fallback without Gate C: **GRL** (letter) or **JGR: Solid Earth** (article).  
Do not submit to Nature-family on SoCal maps alone.

---

## One-sentence claim (current strongest version)

After removing distance, depth, magnitude, mapped Quaternary-fault proximity and station effects, southern California retains two opposite-sign, same-station-validated path-midpoint anomalies in an interpretable coda residual \(\beta_{\mathrm{res}}\); about half of each anomaly survives removal of a classical single-parameter coda-decay proxy, and multipath vs slow-coda still separate inside Qc terciles — i.e. the frozen 5-class taxonomy is not a re-wrapping of scalar Q.

## Physical hypothesis (falsifiable)

| cell | location | \(\Delta\beta\) (same-station) | hypothesis | falsifier |
|---|---|---|---|---|
| **#2** | Salton Trough / S. SAF–Imperial (−116.38, 33.38) | −0.032, 20 stations, 80% sign agree, \(d=-0.54\) | Fluid-rich / thick-sediment paths raise *apparent* attenuation **beyond** fault-distance + station + classical Qc expectation | Independent Qc/Vs cube shows #2 is only a low-Q lobe already mapped at the same scale; or same-station \(\Delta\beta\) vanishes after path-midpoint Vs30 / basin depth is removed |
| **#5** | E. Transverse Ranges (−116.38, 33.88) | +0.025, 15 stations, \(d=+0.68\) | Crystalline mountain-block paths ring longer than the *same stations’* basin paths — a scattering/geometry facies, not higher bulk Q alone | #5 collapses into a high-Qc / high-Vs30 mountain blob with no leftover \(\beta\) residual; or waveform envelopes do not stay energetic later |

If #2 and #5 merely retrace published low-Q vs high-Q provinces, this is a methods paper.  
If they **split** a single Qc/Vs contour — same scalar Q, different facies / opposite \(\beta_{\mathrm{res}}\) — that is the Nature-family fact.

---

## What is already locked (do not reopen)

1. Frozen run28 Huygens picker + ceiling 5-class taxonomy (no re-thresholding after seeing maps).
2. STEAD SoCal fixed-station panel, dist 20–120 km, \(n=8294\); structure-model \(R^2(\beta)=0.27\).
3. Same-station path-midpoint test (in-cell vs out-of-cell + 0.5-cell buffer).
4. Waveform envelope examples: #2 decays faster, #5 persists later (qualitative).
5. **Classical Qc proxy test** (\(n=5000\)): Pearson \(r(\beta_{\mathrm{res}}, q_c)=0.64\).
   - #2 \(\Delta\beta\): −0.030 → **−0.014 after removing \(q_c\)**
   - #5 \(\Delta\beta\): +0.040 → **+0.019 after removing \(q_c\)**
   - Inside every Qc tercile, slow_coda median \(\beta_{\mathrm{res}}\) > multipath.
6. PNW / Alaska residual maps exist; **do not quote composite anomaly scores** (slow_coda near-zero inflates z). Quote \(\beta\) / same-station \(\Delta\beta\) only.

## Gate status

| gate | requirement | status | next action |
|---|---|---|---|
| **A. Independent structure overlay** | Published Qc / Vs at #2/#5 | **Closed.** Berg Vs = **SPLIT** (same Vs band, \(\Delta\beta=+0.046\)). Lin QS = **orthogonal / anti-aligned** (\(r=-0.07\); #2 high QS + faster coda decay; leftover Δβ intact). Classical single-trace \(q_c\) still leaves ~half. | None required for Gate A |
| **B. Out-of-region same-station replica** | ≥8 stations, sign-stable \(\Delta\beta\), \(p<0.05\) outside SoCal | **Closed.** St Helens–Cowlitz: 8 stn, \(\Delta\beta=-0.014\), 88% agree. Jackknife **STABLE** (drop top 1–10 source bins, cap 10–40/bin, year LOO, \(M\ge1.5\) strengthens to −0.039). Seattle basin opposite sign \(\Delta\beta=+0.045\). | None |
| **C. Specific, new physical fact** | One sentence a reviewer cannot rewrite as “basins attenuate / mountains ring” | **Writable.** #2/#5 share Berg Vs but opposite \(\beta\); Salton vs Seattle are opposite-sign basins; St Helens replicates Salton. | Freeze the dossier claim sentence |
| **D. Falsifiable prediction** | Pre-register a held-out cell or a new network | **Written** in manuscript: new SCSN Salton-cell paths \(\Delta\beta\le-0.02\); Seattle \(\Delta\beta\ge+0.02\). | Keep frozen |

Alaska deep-slab paths are a **stress test**, not the replica.

---

## Paper architecture (4 display items)

**Fig. 1 — Taxonomy is physical, not a cluster label.**  
STEAD 5-class pie + \(\rho(\tau)\) characters + example waveforms (existing HNF figure).  
Caption must say: classes assigned by frozen interpretable ceilings, not k-means.

**Fig. 2 — Structure-expectation residual map, SoCal.**  
\(\beta_{\mathrm{res}}\) after log-dist + depth + mag + log1p(fault dist) + station FE.  
Mark #2 (faster) and #5 (slower). Inset: same-station \(\Delta\beta\) forest plot.

**Fig. 3 — More than scalar Q.**  
(a) \(\beta_{\mathrm{res}}\) vs classical \(q_c\) (\(r=0.64\)) + leftover cell \(\Delta\beta\).  
(b) Shape split inside \(q_c\) terciles.  
(c) Independent tomography / Vs30 / basin-depth overlay; #2/#5 vs published contours.  
This is the Nature panel. If (c) fails, Fig. 3 becomes GRL.

**Fig. 4 — Replication + prediction.**  
PNW St Helens (or surviving Cascadia cell) same-station test.  
One pre-registered prediction: e.g. “any new SCSN path with midpoint in #2 will show \(\beta_{\mathrm{res}}<0\) relative to the same station’s non-#2 paths, median \(\Delta\beta \le -0.02\).”

Extended Data: Alaska map, fault-distance nulls, envelope gallery, threshold freeze log.

---

## Abstract draft (≤150 words, update numbers only)

Local earthquake coda is usually collapsed to a single quality factor. We instead assign each STEAD waveform to one of five frozen, interpretable facies using a physics-constrained picker, then remove the expected effects of distance, depth, magnitude, proximity to mapped Quaternary faults, and station. In southern California two path-midpoint domains remain: a faster-decay anomaly at the Salton Trough–Imperial junction (20 stations, same-station \(\Delta\beta \approx -0.03\)) and a slower-decay anomaly in the eastern Transverse Ranges (15 stations, \(\Delta\beta \approx +0.03\)). A classical coda-decay slope explains only about half of each anomaly, and multipath versus slow-coda facies still separate inside matched Qc bands. The same pipeline recovers a Salton-signed faster-decay residual at Mt St Helens–Cowlitz (8 stations) and an opposite, slower-decay residual in the Seattle basin (9 stations) — two sedimentary/volcanic basins that scalar Q would not distinguish. The residuals therefore unmix absorbing versus ringing path regimes. [ Published Qc/Vs overlay sentence when Gate A closes. ]

---

## What we will **not** claim

- That the 5 classes are an earthquake clock or a precursor.  
- That composite PNW/Alaska “scores” are discoveries.  
- That STEAD depth sampling resolves lower crust / mantle (median depth ~8 km).  
- That we have a new fault that QFaults missed — unless tomography + geology close that loop.  
- Any Turkey/Mars-style “new object on the map” without independent closure.

---

## Execution order (no new science until these finish)

1. ~~Berg Vs + Lin QS overlays~~ **Gate A closed.**  
2. Source-jackknife St Helens \(n=911\).  
3. Write NC/NG: direct-wave QS vs coda facies as the central contrast (Lin & Jordan already say these disagree). GRL parallel if you want a faster letter.

## Suggested cover letter angle (if Gates close)

“Seismology has high-resolution velocity models and scalar attenuation maps, but not a taxonomy of *how* a path rings. We show that an interpretable waveform facies residual is spatially coherent, station-controlled, only partly explained by classical Qc, and [replicated / splits a published Q contour]. This is a new observable for crustal path structure, not a new picker benchmark.”
