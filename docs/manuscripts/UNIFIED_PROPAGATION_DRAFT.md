# Local propagation as a unified physical computational primitive: from Huygens seismic facies to cross-domain mechanism coordinates

**Status:** superseded by **`UNIFIED_PROPAGATION_FULL.md`** (Table 1 corrected from `scripts/sn-article.tex`)  
**Do not use Table 1 below** — P MAE 0.021 was from an older snapshot.

---

## Display items (locked)

| ID | Role |
|--|--|
| **Fig. 1** | Concept: \(P(X;G,\Theta,\alpha)\); \(\Theta\) axes; domain landing sites; mini \(\Theta(\lambda)\) |
| **Fig. 2** | Seismic depth: five facies + SoCal \(\beta_{\mathrm{res}}\) + same-station + envelope stacks (**merged** former facies Figs 1–2) |
| **Fig. 3** | Cross-domain \(\Theta^\star\) + \(\Theta(\lambda)\) continuum (wave→damped-wave→diffusion) |
| **Fig. 4** | α gates: EEG CF + LEMON; SST RDG closure; EEG RDG partial contrast |
| **Table 1** | **Full STEAD in-domain head-to-head vs EQTransformer / PhaseNet** |
| **Table 2** | Cross-domain summary: \(\Theta^\star\), key probes, α closure grade |

Extended Data: independence vs \(Q_c/V_S/Q_S\); Cascadia replication/jackknife; special-case verify; physics-map caveat; clinical exploratory; full hyper-parameters.

---

## Abstract (draft, ~180 words)

Complex spatiotemporal systems are often forecast with black-box predictors whose internal propagation rules are not identifiable. We formulate local evolution as a unified physical computational primitive \(\hat{X}=P(X;G,\Theta,\alpha)\), in which wave-like, instantaneous and diffusive couplings are mechanism coordinates \(\Theta\) rather than unrelated models, forecast error is only a probe, and \(\alpha\) is a local deviation from a domain coordinate \(\Theta^\star\). On STEAD seismology, the preferred coordinate is Huygens/wave. Under that prior we assign each local-earthquake waveform to one of five frozen, interpretable coda facies and show that a structure residual \(\beta_{\mathrm{res}}\) maps path domains that are not reducible to scalar coda \(Q_c\). The same skeleton is competitive with EQTransformer and PhaseNet on the STEAD test split and improves P-wave timing (P MAE 21 ms vs 46 ms / 72 ms). Scalp EEG and NOAA SST select different coordinates (instantaneous; graph diffusion), and a telegrapher-inspired path \(\Theta(\lambda)\) exhibits a damped-wave intermediate between wave and diffusion. Finally, \(\alpha\) is geometry-bound under EEG counterfactuals and admits full frozen reinjection closure on SST (partial on EEG range-dependent gain). Local propagation is therefore an identifiable, cross-domain computational object—with its deepest empirical thickness in seismic Huygens facies.

---

## Introduction

[Problem] Forecasting papers maximize next-step error; seismology compresses coda into scalar \(Q\); neuroscience and climate sensor networks lack a shared, falsifiable language for “how information moves.”

[Gap] Without a matched-budget operator family, one cannot tell whether a domain prefers delayed wave propagation, instantaneous coupling, or diffusion—nor whether a fitted deviation field is mechanism or fit residue.

[Approach] We treat recursive local propagation—local state → local response → neighbourhood messages \((A_{ij},\tau_{ij})\) → secondary sources—as one primitive \(P(X;G,\Theta,\alpha)\) (Fig. 1). Named regimes are corners of \(\Theta\); \(\alpha\) multiplies \(K(\Theta^\star)\).

[Contributions]
1. Seismic thickness: Huygens corner → five facies → path residuals; **Table 1** timing vs EQT/PhaseNet as usefulness evidence under the same skeleton.
2. Cross-domain coordinates + continuous \(\Theta(\lambda)\) (Fig. 3).
3. α counterfactual / reinjection gates (Fig. 4).

---

## Results

### A Huygens coordinate for seismology, five coda facies, and competitive timing

Under matched budgets, STEAD multi-station envelopes prefer a delayed wave/Huygens prior over free adjacency and lag-free alternatives (Methods). From the same physics-constrained picker we form frozen ceilings on onset sharpness, coda slope and delay-time density \(\rho(\tau)\), assigning each STEAD waveform to one of five facies—**impulsive fast-\(Q\)**, **emergent**, **multipath**, **slow coda**, **standard**—without post-hoc clustering (Fig. 2a).

After removing log-distance, depth, magnitude, mapped-fault proximity and station effects, the leftover coda residual \(\beta_{\mathrm{res}}\) at path midpoints in southern California yields opposite-sign domains that pass same-station tests: faster decay near the Salton–Imperial junction and longer ringing in the eastern Transverse Ranges (Fig. 2b–d). Classical single-trace \(q_c\) shares only part of this geography; within \(q_c\) terciles, slow-coda and multipath facies still separate; published upper-crust \(V_S\) and direct-wave \(Q_S\) do not absorb the contrast (Extended Data). The same frozen ceilings recover a Salton-signed residual at Mt St Helens–Cowlitz and an opposite-signed residual in the Seattle basin (Extended Data).

**Usefulness under the same skeleton.** On the STEAD EQTransformer test split (\(n=126{,}566\); detection/pick thresholds and tolerance as in Methods), HNF (run28) matches EQTransformer on detection/P F1 within ~0.5–1% absolute, exceeds PhaseNet on P/S F1, and improves **P-wave timing by roughly 2× vs EQTransformer and 3× vs PhaseNet** (Table 1). S MAE remains comparable to EQTransformer. We read Table 1 as evidence that an interpretable Huygens-type prior is not purchased at a large timing cost—not as a claim of uniform state-of-the-art across all transfer settings.

### Cross-domain mechanism coordinates and a continuous dynamical space

Repeating matched-budget ablations, scalp EEG prefers instantaneous coupling and NOAA SST sensor networks prefer graph diffusion (Fig. 3a). Evaluating the unified operator at named corners recovers legacy kernels (exact/unified parity) and reproduces these preferences as empirical \(\Theta^\star\) instances (Fig. 3b). A telegrapher-inspired path \(\Theta(\lambda)\) with \(\lambda=0\) at wave and \(\lambda=1\) at diffusion shows monotone lag collapse and a mid-path damping peak labelled damped-wave (Fig. 3c,d; `dynamical_space_pass`). Thus regimes form a connected mechanism space, with STEAD, EEG and SST as distinct landings (Table 2).

### α as local deviation from domain \(\Theta^\star\)

Under EEG’s instantaneous \(\Theta^\star\), a low-dimensional distance-structured \(\alpha\) is necessary relative to \(\alpha\equiv 1\) and fails under wrong-geometry and edge-permutation counterfactuals; an independent LEMON cohort reproduces the mechanism pattern (Fig. 4a,b). On SST under graph diffusion, a mined range-dependent gain freezes on held-out seeds and passes named-vs-wrong reinjection closure (Fig. 4c). The same RDG family only partially tracks free \(\alpha\) on EEG (Fig. 4d), so closure depth is regime-dependent rather than universal (Table 2).

---

## Discussion

[Thickness] One primitive → deepest evidence in seismic Huygens facies + Table 1 timing → other domains as coordinates → α gated where possible.

[What we do not claim] Universal Huygens; clinical EEG biomarker; OBS/transfer SOTA; five facies as precursors; multi-kernel mixture.

[Limits] Taxonomy gates T2–T4 open; STEAD unified-operator re-run optional; SST physics map is sensor-graph scale.

---

## Methods (outline only — expand later)

**Unified operator.** \(A(\Theta)\), \(\tau(\Theta)\), local response \(R\), \(\alpha\); special cases and \(\Theta(\lambda)\) as in `unified/`.

**STEAD timing (Table 1).** EQTransformer STEAD test split; tol = 0.5 s; pick_th = 0.3; det_th = 0.5; metrics det/P/S F1 and pick MAE (s). HNF run28 checkpoint; SeisBench EQTransformer / PhaseNet STEAD weights. Artifact: `outputs/paper_stead_full_test_compare/`.

**Facies & \(\beta_{\mathrm{res}}\).** Frozen ceilings (peak 1.0; coda −0.198/−0.113; onset 0.877/0.733); ridge structure expectation; same-station test; Berg/Lin overlays; Cascadia box — as in `nc_coda_facies_path_domains.md`.

**Cross-domain / α.** Matched-budget ablations; empirical instances; EEG harden + LEMON; SST RDG H1c′; graded EEG RDG.

---

## Table 1 | STEAD in-domain benchmark versus EQTransformer and PhaseNet

Head-to-head on the STEAD EQTransformer **full test** split (\(n=126{,}566\)). Tolerance 0.5 s; pick threshold 0.3; detection threshold 0.5. MAE in seconds. Bold = best in column.

| Model | Params (nominal) | det F1 | P F1 | S F1 | P MAE (s) | S MAE (s) |
|--|--:|--:|--:|--:|--:|--:|
| **HNF (run28)** | ~192k | 0.9986 | 0.9842 | **0.9756** | **0.021** | 0.088 |
| EQTransformer | ~377k | **0.9989** | **0.9897** | 0.9731 | 0.046 | 0.088 |
| PhaseNet | ~268k | 0.9969 | 0.9512 | 0.9618 | 0.072 | **0.080** |

*Source:* `outputs/paper_stead_full_test_compare/` · README canonical full-test table.  
*Reading:* F1 is neck-and-neck with EQTransformer; the standout is **P-wave timing** (21 ms vs 46 ms / 72 ms). PhaseNet is slightly better on S MAE at lower P F1. Transfer/OBS settings are not claimed here (see Extended Data / prior reports).

---

## Table 2 | Domain mechanism coordinates and α closure (summary)

| Domain | \(\Theta^\star\) | Primary evidence | α / closure grade |
|--|--|--|--|
| STEAD seismology | wave / Huygens | Regime preference; five facies; \(\beta_{\mathrm{res}}\) path domains; Table 1 timing | Weak geometry-sensitive α (Phase-1); not TT residuals |
| Scalp EEG | instantaneous | Matched ablation; empirical instance; harden + LEMON CF | Geometry-bound α **strong**; RDG reinjection **partial–weak** |
| NOAA SST | graph_diffusion | Matched ablation; empirical instance | RDG frozen reinjection **full** (H1c′); physics map with scale caveat |
| Continuum | \(\Theta(\lambda)\): wave→damped-wave→diffusion | `dynamical_space_pass`; mid-λ damping bump | — |

---

## Figure captions (short)

**Fig. 1 | Unified local-propagation primitive.** (a) Recursive local sources. (b) \(P(X;G,\Theta,\alpha)\). (c) Mechanism axes and domain landings. (d) Mini \(\Theta(\lambda)\) phase strip.

**Fig. 2 | Seismic Huygens facies and path residuals.** (a) Five frozen facies. (b) SoCal \(\beta_{\mathrm{res}}\) map. (c) Same-station \(\Delta\beta\). (d) Envelope stacks.

**Fig. 3 | Cross-domain coordinates and continuum.** (a) Regime rankings. (b) Exact/unified parity. (c–d) \(\Theta(\lambda)\) lag, damping, impulse fingerprint.

**Fig. 4 | Validating α.** (a) EEG CF gates. (b) LEMON. (c) SST RDG closure. (d) EEG vs SST RDG depth.

---

## Next production steps

1. Expand Intro/Discussion to full prose; paste facies numbers from NC draft into Results §1.  
2. Build Fig. 2 mega-panel from existing taxonomy + SoCal assets.  
3. Script Fig. 3–4 from unified / harden / SST JSON.  
4. Format Table 1/2 for journal LaTeX.
