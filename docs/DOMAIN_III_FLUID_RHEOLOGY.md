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

## Boltzmann memory track (R0–R1 + aniso SOTA)

### R0 isotropic ablation
Suite board: `outputs/rheo/suite_final/BOARD.md`  
Best isotropic ckpt: `outputs/rheo/best_memory/best.pt` (**r0_k2_full**)

| Model | stress_rel | λ_rel | G_rel | score |
|-------|----------:|------:|------:|------:|
| **r0_k2_full** (best) | **0.0033** | **0.0001** | **0.0002** | **0.0034** |
| r1_k2_aniso | 0.0038 | ≈0 | ≈0 | 0.0038 |
| r0_k2_freq | 0.0033 | 0.0041 | 0.0017 | 0.0058 |
| r0_k2_stress_only | 0.0059 | 0.0706 | 0.0286 | 0.0498 |
| r0_k3_full | 0.0288 | 0.1144 | 0.0219 | 0.0925 |
| r0_k1_stress_only | 0.0470 | 0.1665 | 0.1325 | 0.1700 |

### R1 anisotropic vs domain SOTA (**tuned fair board**)
Board: `outputs/rheo/domain_sota_tuned/BOARD.md`  
**Master zoo (all boards merged):** `outputs/rheo/interpret_mine/MASTER_BOARD.md`  
**Knowledge cards:** `outputs/rheo/interpret_mine/KNOWLEDGE_CARDS.md`  
**Journal panel (a–d):** `docs/figures/rheo/rheo_journal_memory_sota.{png,pdf}`  
(c = multi-step GT | PNF | |RhINN−GT| maps with external \(\sigma\) / \(|\Delta\sigma|\) colorbars)

| Model | stress_rel ↓ | params | Notes |
|-------|-------------:|-------:|-------|
| classical_prony_nls_tuned | **0.0033** | 11 | rheometry NLS (grid on nfev/init) |
| **pnf_aniso** | **0.0033** | **11** | lr/param_weight tuned |
| sparse_prony_euclid_tuned | 0.0147 | 33 | lib/L1/two-stage tuned |
| rhinn_tuned (mech_encode) | 0.0177 | 8.7k | mode/hidden/phys/lr tuned |

**Takeaway:** after val tuning of all methods, PNF ties classical NLS and remains
ahead of tuned EUCLID/RhINN. Untuned RhINN (0.78) was not a fair comparison.

**Interpretability mined from PNF:** recovered \(\lambda\approx[0.50,4.99]\) vs GT \([0.5,5]\);
channel spectra \(G_{11}/G_{22}\) (journal b); startup-shear case recovers anisotropic \(\sigma\) (journal c).

| File | Role |
|------|------|
| `hnf/rheo_memory.py` | `PronyBoltzmannKernel` (λ_k, G_k / A_k, G_∞) |
| `hnf/rheo_baselines.py` | isotropic / diagonal Prony, LSTM, TCN, FIR |
| `hnf/rheo_domain_sota.py` | classical NLS / EUCLID-lite / RhINN |
| `hnf/rheo_synth.py` | startup / oscillatory / multi-step protocols |
| `hnf/rheo_dataset.py` | fixed-material dataset + `RheoMemoryModel` |
| `tools/train_rheo_memory.py` | identify shared Prony spectrum |
| `tools/run_rheo_aniso_sota.py` | full aniso train + SOTA board |
| `tools/tune_rheo_domain_sota.py` | fair val-tune → test-once domain board |
| `tools/plot_rheo_journal_figure.py` | master board + knowledge cards + journal a–d |

```bash
# anisotropic full + SOTA board
PYTHONPATH=. python tools/run_rheo_aniso_sota.py \
  --output-dir outputs/rheo/aniso_sota_full --dim 2 --n-modes 2
# fair domain SOTA (val tune → test once)
PYTHONPATH=. python tools/tune_rheo_domain_sota.py \
  --output-dir outputs/rheo/domain_sota_tuned
# master board + knowledge cards + journal figure a–d
PYTHONPATH=. python tools/plot_rheo_journal_figure.py
# single aniso train
PYTHONPATH=. python tools/train_rheo_memory.py \
  --output-dir outputs/rheo/memory_r1 --anisotropic --dim 2 --n-modes 2 \
  --param-weight 0.5 --epochs 60
```

## Phased deliverables

| Stage | Data | Success criteria | Publishable artifact |
|-------|------|------------------|----------------------|
| **0** | RACLETTE sparse→dense v | Rel. velocity error vs CFD; ablate sparsity | Reconstruction table (supporting) |
| **R0** | Fixed-material Prony + varied protocols | Recover λ_k, G_k; low σ rel. error | Memory-kernel identification table |
| **R1** | Anisotropic A_k + SOTA board | Beat diagonal/isotropic/LSTM on stress_rel | Aniso SOTA table |
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
