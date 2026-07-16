# Domain III: Sparse Flow → Constitutive Discovery

**Status:** Stage-0 + Stage-1 + RACLETTE Stage-0b **done** (2026-07-16).
Stage-1 test: family_acc≈0.80, η_rel≈0.27. RACLETTE Stage-0b inside-vessel
vel_rel≈0.79 @10% keep (weak). `.pv` via `/usr/bin/python3` + pyvista_zstd
(anaconda 3.8 incompatible).
**Priority:** improve RACLETTE recon or Stage-3 mining.  
**Working name:** HNF Fluid / Rheology track.

## Goal

From **sparse velocity observations** (4D Flow MRI–like), jointly:

1. **Forward:** reconstruct / predict denser flow fields  
2. **Inverse:** recover constitutive parameters (e.g. viscosity η, relaxation time λ, power-law index)  
3. **Discover:** screen HNF kernel / latent signals that may indicate constitutive structure beyond a chosen base model family  

Target narrative: one sparse-observation → Physics Decoder → interpretable constitutive pipeline, with publishable accuracy tables and a knowledge-mining report. Claims of “new rheology” stay at **hypothesis / residual analysis**, not premature law announcement.

## Why it fits HNF

| Seismic / Zhizi | EEG (Domain II) | Fluid (Domain III) |
|-----------------|-----------------|--------------------|
| Sparse waveforms | Sparse multi-channel EEG | Sparse 4D Flow voxels / slices |
| Physics Decoder → vp/vs | Classifier / transfer head | Physics Decoder → (η, λ, …) + optional stress |
| Kernel γ,ω,c ↔ wave physics | Kernel ↔ spectral / disease proxies | Kernel ↔ shear-rate / viscoelastic structure |
| Knowledge mining on γ→pick→vp | ROC / ρ(t) group contrasts | Kernel ↔ strain-rate maps; residual vs base constitutive |

Reuse first: `PhysicsDecoder` pattern (`hnf/physics_decoder.py`), frozen or lightly tuned multi-scale HNF encoder, travel-time–style **physics losses** replaced by **momentum / constitutive residual losses**.

## Data prep (can download early)

| Item | Need now? | Size | How |
|------|-----------|------|-----|
| RACLETTE **repo** (loaders/docs) | yes | ~0.5 GB | already under `external_data/raclette/` |
| RACLETTE **example volumes** + all metas | yes (Stage 0) | **~13 GB** | `python download_raclette.py` (public WebDAV, no account) |
| RACLETTE **full cohort volumes** (214 subjects × Data) | later optional | **tens–hundreds GB** | same WebDAV / [DOI](https://doi.org/10.3929/ethz-c-000799752); only after Stage 0 works |
| Constitutive synthetic GT | Stage 1 | small–medium | **generated locally** (no download) |
| Real 4D Flow MRI | Stage 2 | TBD | lock one public cohort later; not started |

Log for the example download: `logs/download_raclette.log`  
Output: `external_data/raclette/Tutorials/DataDownload/Downloaded/`

```bash
cd HNF
python download_raclette.py \
  --out-dir external_data/raclette/Tutorials/DataDownload/Downloaded
```

**Note:** official `Tutorials/DataDownload/tutorial_data_download.py` needs `pyvista` etc.; our slim script only needs `webdav4`.

## Data

### A. RACLETTE (synthetic aortic 4D/5D Flow)

- **What it is:** CFD-enhanced **synthetic aortic 4D/5D Flow MRI** for reconstruction / segmentation / hemodynamic benchmarking ([ETH RACLETTE](https://gitlab.ethz.ch/ibt-cmr/publications/raclette)).  
- **What it is good for here:** realistic **sparse / noisy velocity observations** with CFD **velocity ground truth**; MRI-like undersampling protocols.  
- **What it is *not* (by default):** a catalog of non-Newtonian constitutive GT (λ, η(γ̇), …). CFD in such suites is often Newtonian or fixed hemodynamics assumptions.

**Implication:** use RACLETTE for **Stage 0–1 velocity reconstruction & sparse-observation realism**. Do **not** claim constitutive inversion accuracy on RACLETTE unless the release explicitly provides (or we re-simulate) non-Newtonian GT.

### B. Constitutive synthetic GT (required for Stage 1)

Controlled forward solves with **known** constitutive laws, then sparsify:

| Family | Parameters (examples) | Role |
|--------|----------------------|------|
| Newtonian | η | sanity / baseline |
| Carreau / power-law | η₀, η∞, n, λ_Carreau | shear-thinning blood-like |
| Oldroyd-B / Maxwell | η, λ | viscoelastic relaxation |
| Herschel–Bulkley | τ_y, K, n | yield-stress (optional later) |

Sources: in-house differentiable or offline CFD; literature RhINN / PINN rheology synthetics as external baselines (not mandatory deps).

### C. Real 4D Flow MRI (Stage 2)

Lock **one** public cardiac / vascular 4D Flow cohort before coding the real pipeline (examples to evaluate at kickoff):

- Challenge / open cardiac 4D Flow releases used in reconstruction papers  
- Institutional data only if licensing allows paper release  

**Cross-checks (physiology, not GT λ):** hematocrit–viscosity trends, Newtonian vs non-Newtonian residual under resting vs high-shear regimes, literature ranges for whole-blood apparent viscosity. Fail closed if “inferred λ” contradicts known shear-rate regimes without explanation.

## Method sketch

```text
sparse v(x,t)  [+ optional magnitude images]
        │
        ▼
Channel / spacetime embed  (MRI analogue of 19-ch EEG / 3C seismic)
        │
        ▼
Multi-scale HNF encoder  (reuse principle=huygens_fresnel / multi-scale)
  → ρ(·), kernel summary (γ, ω, c), latent field features
        │
        ▼
Physics Decoder (fluid head)
  → constitutive params θ  and/or dense v̂, optional stress τ̂
        │
        ├── L_data: sparse velocity mismatch
        ├── L_phys: momentum / continuity residual (proxy or weak form)
        ├── L_const: constitutive residual under base family
        └── L_prior: physiological / positivity / smoothness
```

**Physics Decoder reuse**

- Keep: frozen encoder option, `kernel_summary` → head, soft priors, curriculum (head then light kernel path).  
- Replace: layered Earth TT operator → fluid residual operator (start **2D slice / reduced 3D**, not full 5D CFD).  
- Same failure mode to watch: **weak kernel→θ propagation** (apply Domain-II/Zhizi Stage-A fixes: direct kernel summary + intermediate physics loss).

## Phased deliverables

| Stage | Data | Success criteria | Publishable artifact |
|-------|------|------------------|----------------------|
| **0** | RACLETTE sparse→dense v | Rel. velocity error vs CFD; ablate sparsity | Reconstruction table (supporting) |
| **1** | Constitutive synthetic | Param RMSE / relative error vs GT; Newton vs Carreau vs Oldroyd-B ID | **Main inversion accuracy table** |
| **2** | Real 4D Flow | Params in physiological bands; non-Newtonian beats Newton on held-out v | Real-data case study |
| **3** | Mining | Causal chain: Δkernel → Δshear stats → Δθ / residual; FDR-controlled candidates | Knowledge-mining report (hypotheses only) |

## Metrics

**Inversion (Stage 1):** relative error on each θ; recovery rate under noise / sparsity sweeps; confusion between constitutive families (model selection accuracy).  

**Forward (0–2):** RMSE / NRMSE on held-out velocity; mass-conservation residual.  

**Mining (3):** propagation sensitivity \(\|\partial \theta / \partial \omega\|\); residual energy unexplained by best base family; bootstrap stability — **not** “discovered equation” until symbolic distillation + independent validation.

## Suggested repo layout (when implementation starts)

```text
hnf/fluid_dataset.py      # RACLETTE + synthetic constitutive loaders
hnf/fluid_model.py        # encoder + fluid Physics Decoder head
hnf/fluid_physics.py      # residuals / simple constitutive ops
train_fluid.py / eval_fluid.py / run_fluid_analysis.py
docs/figures/fluid/
external_data/raclette/   # gitignored
external_data/fluid_synth/
```

Do not start these while GPU is dedicated to STEAD run27 / OBS download / EEG first train.

## Do-not list

1. **Do not** treat RACLETTE velocity GT as constitutive-parameter GT.  
2. **Do not** claim “new constitutive law” from kernel boxplots alone.  
3. **Do not** start with full 3D Oldroyd-B CFD inside the training loop — begin with reduced operators.  
4. **Do not** parallel-train Domain III on the same GPU as STEAD picking / EEG first runs.  
5. **Do not** skip Stage 1 synthetic recovery; Stage 2 without GT is otherwise unfalsifiable.  
6. **Do not** conflate hemodynamic quantities (WSS, pressure drop) with constitutive discovery without an explicit mapping.

## Kickoff checklist (pre-coding)

- [ ] Confirm RACLETTE download + license; note CFD fluid assumption  
- [ ] Choose Stage-1 constitutive family set (start: Newtonian + Carreau + Oldroyd-B)  
- [ ] Pick one public real 4D Flow source + physiology priors  
- [ ] Write Stage-1 success numbers (e.g. median rel. err. &lt; 10% on λ, η under sparsity s)  
- [ ] Decide operator: analytic 2D channel / PINN residual / offline CFD labels  

## Relation to other domains

| Track | Role now |
|-------|----------|
| STEAD picking + Physics Decoder (seismic) | Primary; finish run27 / OBS transfer |
| Domain II EEG (`hnf/eeg_*`, `train_eeg.py`) | Next implementation after picking bandwidth frees |
| Domain III fluid (this doc) | Design frozen until EEG Stage-1 or explicit reprioritization |

---

*Last updated: 2026-07-16. Stage-0 synthetic training launched; RACLETTE I/O pending pyvista.*
