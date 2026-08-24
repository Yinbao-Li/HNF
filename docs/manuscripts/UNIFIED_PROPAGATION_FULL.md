# Local propagation as a unified physical computational primitive: from Huygens seismic facies to cross-domain mechanism coordinates

**LaTeX full manuscript:** `scripts/unified-propagation-article.tex`  
(Compile from `scripts/`: `pdflatex unified-propagation-article.tex`. Main Figs 1–4 and Extended Data Figs 1–6 live under `docs/figures/unified/`.)

---

## Display items

| ID | Content |
|--|--|
| **Fig. 1** | Unified primitive \(P(X;G,\Theta,\alpha)\); mechanism axes; domain landings; mini \(\Theta(\lambda)\) |
| **Fig. 2** | Five coda facies + SoCal \(\beta_{\mathrm{res}}\) + same-station controls + envelope stacks |
| **Fig. 3** | Cross-domain \(\Theta^\star\) rankings + \(\Theta(\lambda)\) continuum |
| **Fig. 4** | α validation: EEG CF/LEMON; SST RDG closure; EEG vs SST RDG depth |
| **Table 1** | Full STEAD test vs EQTransformer / PhaseNet (**sn-article numbers**) |
| **Table 2** | Domain \(\Theta^\star\) and α-closure summary |

**Extended Data (SI; regenerate with `tools/plot_unified_ed_figures.py`):**

| ED | Label | Content |
|--|--|--|
| 1 | `edfig:faciesgeo` | Facies geography map + morphology-class characters |
| 2 | `edfig:independence` | \(\beta_{\mathrm{res}}\) vs \(q_c\) / \(V_S\) / \(Q_S\) + leftover bars |
| 3 | `edfig:cascadia` | Cascadia residual maps + SoCal / St Helens jackknives |
| 4 | `edfig:classical` | Catalog-timed vs probe-timed coda control (\(n=4000\)) |
| 5 | `edfig:unified` | \(\Theta(\lambda)\) lag, damping proxy, phase sequence |
| 6 | `edfig:sstphys` | SST RDG \(g(\hat r)\) + near/mid/far + log-km scale caveat |

Corner recovery of unified named backends: Supplementary Table `stab:corners` (not a figure panel).
Clinical associations remain Supplementary Note only.

---

## Abstract

Complex spatiotemporal systems are often forecast with black-box predictors whose internal propagation rules are not identifiable. We cast local evolution as a unified physical computational primitive \(\hat{X}=P(X;G,\Theta,\alpha)\), in which wave-like, instantaneous and diffusive couplings are mechanism coordinates \(\Theta\) rather than unrelated models; forecast error is only a probe; and \(\alpha\) is a local deviation from a domain coordinate \(\Theta^\star\). On the STEAD seismic benchmark, the preferred coordinate is Huygens/wave. A physics-constrained probe with \(192\mathrm{k}\) parameters matches EQTransformer on detection F1, improves S-wave F1, and achieves the lowest P-wave timing error (P MAE \(0.0169\,\mathrm{s}\) versus \(0.0462\,\mathrm{s}\) / \(0.0723\,\mathrm{s}\) for EQTransformer / PhaseNet; Table 1). Under the same prior we assign each local-earthquake waveform to one of five frozen, interpretable coda facies and map a structure residual \(\beta_{\mathrm{res}}\) that separates path domains not reducible to scalar coda \(Q_c\). Scalp EEG and NOAA SST select different coordinates (instantaneous; graph diffusion), and a telegrapher-inspired path \(\Theta(\lambda)\) exhibits a damped-wave intermediate between wave and diffusion. Finally, \(\alpha\) is geometry-bound under EEG counterfactuals and admits full frozen reinjection closure on SST (partial on EEG range-dependent gain). Local propagation is therefore an identifiable cross-domain computational object, with its deepest empirical thickness in seismic Huygens facies.

---

## Introduction

Scientific machine learning faces a persistent fork. Purely data-driven networks can score well on detection and next-step prediction, yet their representations rarely encode known causal propagation and are difficult to trust as instruments of discovery. Physics-informed penalties improve consistency only approximately. Hybrid “explainable” heads often attach post-hoc probes to opaque backbones, so the explanation is correlative rather than constitutive of the computation.

A sharper requirement is needed: **known local propagation physics should enter as a differentiable computational primitive**—a readable Green’s-function-like operator that interrogates data—while still allowing domains to differ in *which* propagation regime they select. Seismology already suggests that delayed wave physics matters for multi-station envelopes and picks, but coda “how a path rings” is still usually collapsed to a scalar quality factor \(Q_c\). Scalp EEG and climate sensor networks raise the dual question: do they share the same delayed-wave language, or different coordinates of one family?

Here we formulate recursive local evolution

\[
\text{local state}\rightarrow\text{local response}\rightarrow\text{neighbourhood messages }(A_{ij},\tau_{ij})\rightarrow\text{secondary sources}
\]

as one operator \(\hat{X}=P(X;G,\Theta,\alpha)\) (Fig. 1). Geometry \(G\) supplies distances and sampling; structural parameters \(\Theta\) continuously interpolate delay, spatial kernel family, normalisation and travel-time damping; \(\alpha\) multiplies the kernel at a domain coordinate \(\Theta^\star\). Wave/Huygens, instantaneous coupling, continuum diffusion and graph diffusion are named corners of \(\Theta\), not three unrelated architectures. A one-parameter telegrapher-inspired path \(\Theta(\lambda)\) connects wave (\(\lambda=0\)) to diffusion (\(\lambda=1\)) through a damped-wave intermediate.

We make three linked contributions. First, on STEAD we show that a Huygens-type probe is competitive with EQTransformer and PhaseNet on the full test set while remaining physically structured (Table 1), and that the same probe supports a **five-class frozen coda facies taxonomy** whose structure residual maps path domains beyond scalar \(Q_c\) (Fig. 2). Second, matched-budget ablations place scalp EEG and NOAA SST at different \(\Theta^\star\), and \(\Theta(\lambda)\) behaves as a continuous dynamical space (Fig. 3; Table 2). Third, \(\alpha\) is validated as a geometry-bound deviation from domain \(\Theta^\star\) by counterfactuals and, where successful, frozen reinjection (Fig. 4).

We do **not** claim universal Huygens physics, a clinical EEG biomarker, or that the five facies are precursors. Forecast metrics are usefulness evidence under an interpretable prior, not the scientific endpoint.

---

## Results

### A Huygens coordinate on STEAD: competitive timing under a physics-constrained probe

We evaluate the multi-scale Huygens–Fresnel probe (PNF/HNF run family; \(192\mathrm{k}\) parameters) on STEAD under the standard \(0.5\,\mathrm{s}\) tolerance protocol used for EQTransformer. Table 1 reports full test-set detection and picking against EQTransformer (\(\sim 377\mathrm{k}\) parameters) and PhaseNet (\(\sim 269\mathrm{k}\)).

PNF **matches EQTransformer on detection F1** (\(0.9989\)), **improves S-wave F1** (\(0.9770\) vs \(0.9731\) / \(0.9618\)), and achieves the **lowest P-wave MAE** (\(0.0169\,\mathrm{s}\) on the discrete \(800\)-bin / \(60\,\mathrm{s}\) grid; Methods), roughly \(2.7\times\) tighter than EQTransformer (\(0.0462\,\mathrm{s}\)) and \(4.3\times\) tighter than PhaseNet (\(0.0723\,\mathrm{s}\)). It does not dominate every cell: P-wave F1 (\(0.9852\)) trails EQTransformer (\(0.9897\)), and S-wave MAE (\(0.0878\,\mathrm{s}\)) trails PhaseNet (\(0.0801\,\mathrm{s}\)). Precision/Recall in Table 1 shows strong recall across heads; gaps to EQTransformer are driven mainly by precision (false-positive suppression), consistent with an unconstrained transformer trading physical structure for slightly tighter precision.

We read Table 1 as evidence that an interpretable Huygens-type computational prior is **not purchased at a large timing cost**—and on P timing is markedly better—rather than as a claim of uniform state-of-the-art across all transfer settings (OBS/adapt results are reported elsewhere and are not used as primary claims here).

### Five frozen coda facies and path-structure residuals

From the same physics-constrained picker we form causal-chain observables (coda slope, onset sharpness, delay-time density \(\rho(\tau)\)) and assign each STEAD local-earthquake waveform to one of five facies using **ceilings frozen before regional maps were inspected**: impulsive fast-\(Q\), emergent, multipath, slow coda, and standard (Fig. 2a). Classes are not \(k\)-means labels. Multipath traces show multi-lobe \(\rho(\tau)\); slow-coda traces keep energy later after \(S\); impulsive-fast-\(Q\) traces combine sharp onsets with steep coda decay.

Facies geography on the STEAD test catalogue (noise excluded) is organised at continental scale (Cramér’s \(V\approx 0.37\), FDR-adjusted \(p<0.001\)): impulsive fast decay enriches in the western United States and northeast Pacific, whereas slow coda concentrates in Central America and the western Pacific (Extended Data).

After distance-detrending coda slope to \(\beta\) and removing a ridge expectation on \(\log_{10}\) distance, depth, magnitude, log-distance to nearest Quaternary fault, and station fixed effects, the leftover \(\beta_{\mathrm{res}}\) is mapped at path midpoints. In southern California (fixed stations; \(20\)–\(120\,\mathrm{km}\)), two opposite-sign domains pass same-station tests (Fig. 2b–d): Salton–Imperial midpoints decay faster than the same stations’ other paths (site-median \(\Delta\beta\approx -0.032\); \(20\) stations), whereas eastern Transverse Ranges midpoints ring longer (\(\Delta\beta\approx +0.025\); \(15\) stations). Envelope stacks confirm the signs.

Independence checks (Extended Data) show that classical single-trace \(q_c\) correlates with \(\beta_{\mathrm{res}}\) (\(r=0.64\)) but residualising on \(q_c\) leaves coherent cell anomalies, and multipath versus slow-coda still separate inside matched \(q_c\) terciles. Berg et al. upper-crust \(V_S\) is nearly uncorrelated (\(r=-0.08\)) and the two cells occupy the same \(V_S\) band with opposite \(\beta_{\mathrm{res}}\). Lin & Jordan direct-wave \(Q_S\) is likewise near-orthogonal (\(r=-0.07\)) and locally anti-aligned with naive “high \(Q\) ⇒ slow coda” intuition. The identical frozen pipeline on Cascadia recovers a Salton-signed residual near Mt St Helens–Cowlitz and an opposite ringing residual near Seattle (Extended Data; presented as out-of-region consistency examples).

**Classical-coda control.** Because \(\beta_{\mathrm{res}}\) is computed from a raw-envelope slope in an \(S\)-timed window, we recompute the identical slope using STEAD catalog \(S\) picks on \(n=4000\) SoCal traces. Catalog-timed and probe-timed slopes agree closely (\(r=0.96\)); \(\beta_{\mathrm{res}}\) maps agree at \(r=0.93\); same-station cell signs are essentially unchanged. Adding probe-specific latents yields \(\Delta R^{2}\approx 0\) for predicting probe \(\beta_{\mathrm{res}}\) from classical coda features. We therefore report the spatial leftover as **classical coda geography that the Huygens probe reproduces and organises into frozen facies**, not as a uniquely neural residual that classical envelopes cannot see. The scientific gain is the **interpretable five-mode taxonomy and its use as a path observable under a Huygens computational prior**, together with Table 1 showing that prior remains task-competitive.

### Cross-domain mechanism coordinates and a continuous dynamical space

Under matched training budgets, local-response capacity and geometry controls, scalp EEG prefers **instantaneous** coupling (lag \(\equiv 0\)) over delayed Huygens, whereas NOAA SST sensor networks prefer **graph diffusion** (Fig. 3a; Table 2). Evaluating the unified operator at named corners recovers legacy kernels (exact versus unified backends in parity) and reproduces these preferences as empirical \(\Theta^\star\) instances (Fig. 3b). STEAD’s Huygens landing, EEG’s instantaneous landing and SST’s graph-diffusion landing therefore appear as distinct coordinates of one family rather than three unrelated model classes.

A telegrapher-inspired path \(\Theta(\lambda)\) with delay \(\delta=(1-\lambda)^{2}\), heat mix \(\eta=\lambda^{2}\) and travel-time damping \(\zeta=4\lambda(1-\lambda)\) connects wave (\(\lambda=0\)) to diffusion (\(\lambda=1\)). Mean lag collapses monotonically; mid-path damping peaks; phase labels traverse wave-like → damped-wave → diffusive (Fig. 3c,d). This supports the stronger claim that regimes form a **continuous dynamical space**, not only a discrete dictionary.

### α as local deviation from domain \(\Theta^\star\)

We reinterpret \(\alpha\) as a multiplicative deviation from \(K(\Theta^\star)\), not as a universal residual relative to Huygens in every domain. Under EEG’s instantaneous \(\Theta^\star\), a low-dimensional distance-structured \(\alpha\) is necessary relative to \(\alpha\equiv 1\) and degrades under wrong-geometry and edge-permutation counterfactuals; an independent LEMON cohort reproduces the mechanism pattern (Fig. 4a,b). On SST under graph diffusion, a mined range-dependent gain freezes on held-out seeds and passes named-versus-wrong reinjection closure without refitting mechanism parameters (Fig. 4c). The same RDG family only partially tracks free \(\alpha\) on EEG (Fig. 4d), so closure depth is regime-dependent (Table 2). Exploratory clinical associations of EEG leftovers with cognition are deferred to Extended Data and are **not** claimed as biomarkers.

---

## Discussion

This work thickens a single research line. A Huygens-type local propagator is both **task-useful** on STEAD (Table 1) and **scientifically productive** as the coordinate that organises five frozen coda facies and path residuals (Fig. 2). Cross-domain ablations show that the same primitive lands elsewhere at instantaneous and graph-diffusion coordinates, and \(\Theta(\lambda)\) supplies a continuous bridge with a damped-wave phase (Fig. 3). Validated \(\alpha\) then measures local departure from those landings (Fig. 4).

Huygens is therefore the right *interpretable computational principle for seismology* in this study—not a universal law of cortex or ocean. The five facies classify how paths ring under that principle; they are not earthquake clocks or precursors. The classical-coda geography control clarifies that the map leftover is not a mysterious neural-only field, while still elevating facies as an interpretable discrete language for path structure.

Limitations include open taxonomy gates beyond synthetic recovery, incomplete pre-registered spatial replication catalogs for Cascadia, sensor-graph scale caveats for SST physics maps, and the deliberate exclusion of clinical biomarker claims. Future work can extend \(\Theta\) with reaction, advection and richer damped-wave terms without rewriting the training skeleton.

---

## Methods

### Unified operator \(P(X;G,\Theta,\alpha)\)

State \(X\in\mathbb{R}^{N\times T}\) lives on sensors/stations. Geometry \(G\) provides pairwise distances and \(\mathrm{d}t\). The interaction amplitude \(A(G;\Theta)\) blends wave-like \(e^{-\gamma r^{2}}/(r+\varepsilon)\), graph \(e^{-r/\ell}\) and heat-kernel shapes via \((\mathrm{spatial\_mix},\mathrm{heat\_mix})\); optional travel-time damping multiplies \(e^{-\zeta\tau_{\mathrm{geo}}}\). Lags are \(\tau=\mathrm{round}(\delta\cdot\tau_{\mathrm{geo}})\). Local response \(R\) maps self state and aggregated neighbour drive to the next field. Reality uses \(A\odot\alpha\). Named corners recover legacy STEAD/EEG/SST priors (`exact` backend delegates to established kernels; `unified` corners match within numerical tolerance). Code: `hnf/propagation_dynamics/unified/`.

### \(\Theta(\lambda)\) continuum

Telegrapher proxy: \(\delta=(1-\lambda)^{2}\), \(\eta=\lambda^{2}\), \(\zeta=4\lambda(1-\lambda)\). Continuity gates require finite amp/MSE steps, monotone lag decrease on wave→diffusion, and visitation of wave-like and diffusive ends; damped-wave is recorded when mid-path damping peaks (`outputs/propagation_dynamics/unified_theta_lambda_v1/`).

### STEAD timing protocol (Table 1)

Full STEAD test evaluation under tolerance \(0.5\,\mathrm{s}\); pick/detection thresholds as in the EQTransformer protocol. Metrics: detection/P/S F1 with precision/recall, and pick MAE (s). PNF/HNF parameter count \(\sim 192\mathrm{k}\); EQTransformer \(\sim 377\mathrm{k}\); PhaseNet \(\sim 269\mathrm{k}\). **Authoritative numbers are those published in `scripts/sn-article.tex` Table `tab:stead`.** (An earlier repository snapshot `outputs/paper_stead_full_test_compare/` reported a slightly different HNF line, e.g. P MAE \(0.0213\,\mathrm{s}\); that snapshot is **superseded** for this manuscript.)

### Facies ceilings and \(\beta_{\mathrm{res}}\)

Frozen ceilings: peak \(1.0\); coda-fast \(-0.198\); coda-slow \(-0.113\); onset-high \(0.877\); onset-low \(0.733\). \(\beta\) is the residual of coda slope on \(\log_{10}(\mathrm{distance}+1)\). Structure expectation: ridge (\(\lambda=8\)) on \(\log_{10}\) distance, depth, magnitude, \(\log(1+\mathrm{fault\ distance})\), station FE. Same-station test: in-cell versus out-of-cell paths at the same station with buffer; discovery threshold \(\ge 8\) stations and \(p<0.05\). Classical \(q_c\), Berg \(V_S\), Lin \(Q_S\), Cascadia box and jackknives follow `docs/manuscripts/nc_coda_facies_path_domains.md` and `scripts/sn-article.tex`. Catalog-timed classical-coda control as described in Results.

### Cross-domain ablations and α gates

Matched-budget regime ablations (wave / instantaneous / diffusion / graph_diffusion) with shared response freeze policy. Empirical \(\Theta^\star\) instances: `outputs/propagation_dynamics/unified_empirical_instances_v1/`. EEG instantaneous harden and LEMON replication: subject-bootstrap counterfactual gates. SST RDG: mine on one seed, freeze, evaluate held-out seeds (H1c′, \(\ell=0.5\)); no mechanism refit at reinjection. EEG RDG graded closure reported as partial–weak contrast.

### Statistics

Report effect sizes (Cohen’s \(d\)), nonparametric \(p\) where used, and bootstrap intervals for EEG gates. Facies–geography association uses Cramér’s \(V\) with FDR adjustment as in the seismic draft.

---

## Table 1 | STEAD full-test benchmark versus EQTransformer and PhaseNet

**Source:** `scripts/sn-article.tex`, Table `tab:stead` (authoritative).  
For each task, F1 and Precision/Recall (P/R) are reported. MAE in seconds. Best bold; second-best underlined in the LaTeX original.

| Model | \# Params | Det F1 | Det P/R | P F1 | P P/R | S F1 | S P/R | P MAE (s) | S MAE (s) |
|--|--:|--:|--|--:|--|--:|--|--:|--:|
| EQTransformer | 377k | **0.9989** | 0.9992/0.9989 | **0.9897** | 0.9993/0.9794 | 0.9731 | 0.9994/0.9470 | 0.0462 | 0.0874 |
| PhaseNet | 269k | 0.9969 | 0.9969/0.9975 | 0.9512 | 0.9984/0.9093 | 0.9618 | 0.9982/0.9283 | 0.0723 | **0.0801** |
| **PNF** | 192k | **0.9989** | 0.9994/0.9984 | 0.9852 | 0.9942/0.9763 | **0.9770** | 0.9900/0.9643 | **0.0169** | 0.0878 |

*Reading:* Detection tied with EQTransformer; best S F1 and best P timing; P F1 second to EQTransformer; S MAE second to PhaseNet. Physics-structured probe with roughly half EQTransformer’s parameters.

---

## Table 2 | Domain mechanism coordinates and α closure

| Domain | \(\Theta^\star\) | Primary evidence | α / closure |
|--|--|--|--|
| STEAD | wave / Huygens | Table 1; five facies; \(\beta_{\mathrm{res}}\) path domains | Weak geometry-sensitive α (Phase-1); not TT residuals |
| Scalp EEG | instantaneous | Matched ablation; empirical instance; harden + LEMON | Geometry-bound α **strong**; RDG **partial–weak** |
| NOAA SST | graph_diffusion | Matched ablation; empirical instance | RDG reinjection **full** (H1c′) |
| Continuum | \(\Theta(\lambda)\) | wave→damped-wave→diffusion; lag monotone; damping bump | — |

---

## Figure captions

**Fig. 1 | Unified local-propagation primitive.** (a) Recursive local sources. (b) Operator \(P(X;G,\Theta,\alpha)\). (c) Mechanism coordinates and STEAD/EEG/SST landings. (d) Mini \(\Theta(\lambda)\) phase strip (wave → damped-wave → diffusion).

**Fig. 2 | Seismic Huygens facies and path residuals.** (a) Five frozen facies (waveforms / \(\rho(\tau)\)). (b) SoCal \(\beta_{\mathrm{res}}\) map with Salton and Transverse Ranges cells. (c) Same-station \(\Delta\beta\). (d) In-cell versus out-of-cell envelope stacks.

**Fig. 3 | Cross-domain coordinates and continuum.** (a) Matched-budget regime rankings. (b) Exact/unified kernel parity at corners. (c) \(\Theta(\lambda)\): mean lag and damping. (d) Impulse-response fingerprint along \(\lambda\).

**Fig. 4 | Validating α.** (a) EEG counterfactual gates. (b) LEMON replication. (c) SST named RDG reinjection closure. (d) Graded EEG RDG versus SST full closure.

---

## Data and code availability

STEAD as released by Mousavi et al.; SeisBench baselines; USGS QFaults; Berg 2021 and Lin & Jordan 2023 grids as cited in the seismic draft. Unified-operator and α artifacts under `outputs/propagation_dynamics/`. Primary LaTeX numbers for Table 1: `scripts/sn-article.tex`. Analysis code in the HNF repository (`tools/`, `hnf/propagation_dynamics/`).

---

## References (keys to resolve in BibTeX)

Mousavi et al. STEAD; Mousavi et al. EQTransformer; Zhu & Beroza PhaseNet; Berg et al. 2021 \(V_S\); Lin & Jordan 2023 \(Q_S\); Raissi/Karniadakis PINNs; domain EEG/SST dataset citations as in existing drafts.

---

## Supplementary Information / Extended Data

**Authoritative SI lives in LaTeX** (`scripts/unified-propagation-article.tex`, after `\clearpage\onecolumn`):

- ED Figs 1–6 placeholders: facies geography; \(q_c/V_S/Q_S\) independence; Cascadia; classical-coda control; unified continuum/\(\Theta(\lambda)\); SST physics-map caveat
- SI Tables: ceilings; same-station cells; independence leftovers; unified verify snapshots; α gates (EEG +82% / LEMON +67% / SST H1c′ / EEG RDG graded)
- Notes: Physics Decoder (not main claim); exploratory clinical leftover (not biomarker); explicit non-claims; freeze checklist

Main-text display budget remains **4+2**; SI does not count against it.

---

## Change log

- **2026-08-22 (late):** Rewrite per editorial notes: (i) no defensive non-claims dump—intent/limits stated in place; (ii) Θ axes given explicit physical meaning; continuity = operator-path continuity, **not** Lyapunov trajectory stability; (iii) facies = physics-indexed axes + operational validation ceilings (not closed-form physics cut-points, not clustering); classical coda control kept as clarifying contribution, consistent with unified story; (iv) all repo/artifact paths removed from manuscript; (v) main text expanded for readability. Authoritative file: `scripts/unified-propagation-article.tex`.
- **2026-08-22 (evening):** Main LaTeX thickened (facies ceilings/stats, independence leftovers after residualising \(q_c\), Cascadia jackknife numbers, cross-domain MSE/rel_geom, α CF +82%/+64%, LEMON +67%, SST/EEG RDG grades, P/R trade-off). Full SI/ED section added. Markdown here remains a digest; compile LaTeX for the full article+SI.
- **2026-08-22:** Table 1 corrected to `sn-article.tex` (`P MAE 0.0169`, det F1 `0.9989`, S F1 `0.9770`, with P/R). Supersedes README/`paper_stead_full_test_compare` line used in the earlier skeleton. Full prose draft assembled under the 4+2 display budget.
