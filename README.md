# Huygens Neural Field (HNF)

A physics-inspired neural field built on the Huygens principle. A learnable complex kernel models wave-like interactions; the same stack supports sparse field reconstruction, STEAD phase picking, and 1D velocity inversion with a frozen picking backbone (“Zhizi” bridge).

```
Kernel + density design
  → Framework (layers, field reconstruction)
  → STEAD classification & phase picking (run20)
  → 1D travel-time / FWI-lite baselines
  → Zhizi inversion bridge (macro Physics Head)
  → Proof suite (geometry-aware STEAD + baselines + latent plots)
  → Interpretability suite (kernel χ, contrib rows, ablations)
```

| Stage | Artifact | Result |
|-------|----------|--------|
| Picking | `outputs/run20/20_wrongpeak_sharp/best.pt` | det F1 **0.994** / P **0.959** / S **0.949** (~**139k** params) |
| Inversion init | `outputs/zhizi_inversion_bridge_macro/best_physics_head.pt` | Waveform refine win-rate **93.8%** (synth); STEAD geom refine **77.1%** |

Figures below live in [`docs/figures/`](docs/figures/) (copied from completed `outputs/` runs). Inversion recipe details: [`README_ZHIZI_INVERSION.md`](README_ZHIZI_INVERSION.md).

---

## Setup

```bash
cd HNF
pip install -r requirements.txt
```

- Python deps: `torch>=2.0`, `numpy`, `matplotlib`, `pytest`, `tqdm`, `openpyxl`
- Place STEAD under `STEAD/` (~90GB; gitignored)
- Large run products stay in `outputs/` (gitignored); key plots are mirrored to `docs/figures/`
- GPU ≥12GB recommended; picking uses `seq_len=800`; bridge inference often uses `infer_seq_len=600`

```bash
python -c "from hnf import HuygensKernel, HuygensNeuralField, STEADHNFPickingModel; print('ok')"
pytest hnf/tests -q
```

---

## 1. Model design

Huygens kernel (`hnf/kernel.py`):

\[
K_{\text{Huygens}}(x_i,x_j)=\frac{1}{r^2+\varepsilon}\exp(-\gamma r^2)\exp(i\,\omega r)
\]

**Huygens–Fresnel** variant (`--principle huygens_fresnel`): spherical \(1/r\) amplitude, extra \(i\omega/(2\pi)\) phase, and obliquity \(\chi(\theta)=\tfrac12(1+\cos\theta)\) suppressing off-axis secondary sources. Selected via `--principle` on the picking trainer; default remains `huygens`.

| Piece | Role |
|-------|------|
| Complex phase `exp(i ω r)` | Interference / travel-time structure |
| Gaussian envelope `exp(-γ r²)` | Local secondary-source weight |
| Causality + wave speed | Directed temporal propagation |
| Learnable γ, ω, wave_speed | Soft physical adaptation |
| Distance modes: feature / time / hybrid | Field coordinates or waveform time |

Supporting modules:

- **`DensityNet`** (`density.py`) — spatial density ρ(x), Softplus-positive
- **`HuygensWaveLayer` / `HuygensAttention`** (`layers.py`) — stack the kernel in deep models
- **`FastMultipoleMethod`** (`fmm.py`) — far-field acceleration
- **`DeepHuygensKernel`**, **`BayesianHNF`** — deeper / uncertainty variants

In the picking model, **ρ(t)** and kernel wave-speed are **soft conditioners**, not literal crustal density or absolute velocity. Physical `vp/vs` comes from the physics head + refinement.

---

## 2. Field reconstruction

`HuygensNeuralField` solves a kernel regression from sparse observations:

```
(x_obs, y) → K_obs = Re(K(obs,obs))
         → w = (K_obs + αI)^{-1} y
         → field = Re(K(target,obs)) @ w
```

```bash
python example_2d_reconstruction.py
python example_2d_reconstruction.py --field-type vortex --n-obs 200 --train-steps 300
```

Plot helpers: `hnf/visualize.py`. Kernel demos: `hnf/demos.py` (`demo_causality`, `demo_fmm_benchmark`, …).

---

## 3. STEAD: classification → phase picking

### Classification

```bash
python train_stead.py --device cuda
```

Validates Huygens attention on STEAD earthquake / noise waveforms.

### Picking model (`STEADHNFPickingModel`)

Three-component secondary sources → temporal `rho(t)` → Huygens wave blocks (optional noise-cancel branch) → det / P / S heads (envelope-residual pick heads).

Trainer: `train_stead_picking.py`. Orchestration scripts: `run11_stead_picking.py` … `run20_stead_picking.py`.

Design choices retained in the final model:

- Preserve full temporal resolution and stable detection, then push P/S
- Denoise branch primarily for **det**; P/S use **raw** waveform plus denoise cues
- Stage-wise freezing of backbone / det while refining pick heads
- Short low-LR sharp pass with wrong-peak suppression (**run20**)

**Frozen checkpoint**

```text
outputs/run20/20_wrongpeak_sharp/best.pt
  det_f1 ≈ 0.994   p_f1 ≈ 0.959   s_f1 ≈ 0.949   n_params ≈ 139402
```

```bash
python run20_stead_picking.py
python eval_stead_picking.py --checkpoint outputs/run20/20_wrongpeak_sharp/best.pt
python explain_stead_picking.py --checkpoint outputs/run20/20_wrongpeak_sharp/best.pt
```

Dataset: `hnf/stead_picking_dataset.py` (includes `source_distance_km` / `source_depth_km` for real-event geometry).

**Pick threshold sweep**

![Picking threshold sweep](docs/figures/picking_threshold_sweep.png)

*Figure: threshold vs picking metrics on the run20 model (`picking_threshold_sweep.png`).*

---

## 4. 1D inversion baselines

Before the Zhizi bridge, a layered-Earth stack was built and compared end-to-end.

| Component | Module |
|-----------|--------|
| Layered Earth + P/S travel times | `hnf/inversion_1d.py` |
| Gauss–Newton / L-BFGS / Adam | `hnf/inversion_baselines.py` |
| Acoustic FWI-lite | `hnf/acoustic_fwi_1d.py` |
| Synthetic waveforms | `hnf/synth_waveforms_1d.py` |
| Ray paths | `hnf/ray_paths.py` |
| Profile / misfit plots | `hnf/inv_plot.py` |

```bash
python run_inv01_synth_1d.py
python run_inv_full_compare.py
python run_inv_fwi_lite.py
python run_inv05_pick_to_inversion.py
```

**Takeaway:** classical travel-time solvers (esp. GN / L-BFGS) reach the lowest absolute Vp RMSE on synthetic oracles. Waveform FWI-lite improves from a given start model. The Zhizi line therefore targets a **better waveform-inversion initializer**, scored against a standard perturbed start.

![Inversion full comparison](docs/figures/full_comparison.png)

*Figure: multi-method 1D inversion overview (`full_comparison.png`).*

---

## 5. Zhizi inversion bridge

### Pipeline

```
Frozen run20
  → station features: rho(t), envelope, kernel soft scales, P/S picks
  → macro Physics Head: scale / contrast / Vs ratio
  → vp0/vs0 relative to a reference layered model (zero init ≈ reference)
  → short differentiable waveform refine (Route A2) or travel-time refine
```

Code: `hnf/zhizi_physics_head.py`, `zhizi_inversion_bridge.py`, `zhizi_inversion_dataset.py`, `zhizi_inversion_loss.py`.

### Training (converged recipe)

Short training is sufficient; best Val Vp RMSE ≈ **0.277** near epoch 4.

```bash
python train_zhizi_inversion.py \
  --head-mode macro --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/zhizi_inversion_bridge_macro
```

Checkpoint: `outputs/zhizi_inversion_bridge_macro/best_physics_head.pt`.

![Training curves](docs/figures/training_curves.png)

*Figure: validation Vp RMSE, total loss, and unrolled Vp MSE (`training_curves.png`). Best early in the run.*

### Route A2 (synthetic waveform refine)

```bash
python run_route_a2_waveform.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --n-test 32 --fwi-steps 60 --device cuda
```

| Setting | Zhizi + wave VpRMSE | Perturb + wave | Zhizi better |
|---------|---------------------|----------------|--------------|
| 32 events | **0.924** | 0.982 | **93.8%** |
| 64 events | **0.935** | 0.977 | **87.5%** |

One-shot init need not beat a hand perturbation; the macro deformation more often lands FWI-lite in a better basin.

---

## 6. Proof suite (geometry-aware STEAD + baselines + latents)

```bash
python run_proof_suite.py --device cuda --max-events 48 --n-synth 32 \
  --output-dir outputs/proof_suite
```

Full JSON: `outputs/proof_suite/proof_report.json`. Figures below are the same plots shipped in `docs/figures/`.

### STEAD with real epicentral distance / depth (n=48)

| Metric | Zhizi refine | Perturb refine |
|--------|--------------|----------------|
| Mean TT misfit | **3.08** | 11.22 |
| Win rate | **77.1%** | — |
| Wilcoxon (approx.) | p ≈ 3×10⁻⁵ | — |

![STEAD refine scatter](docs/figures/stead_refine_scatter.png)

*Figure: points below the diagonal favor Zhizi after travel-time refine (`stead_refine_scatter.png`).*

![STEAD geometry conditioning](docs/figures/stead_geom_conditioning.png)

*Figure: TT misfit delta vs distance (color = depth) and win-rate by distance bin (`stead_geom_conditioning.png`). Longer ranges show higher Zhizi win rates.*

### Synthetic full baseline compare (n=32)

| Method | Mean Vp RMSE |
|--------|--------------|
| zhizi_wave | **0.924** |
| perturb_wave | 0.982 |
| gn_tt (travel-time oracle) | 0.136 |
| lbfgs_tt | 0.201 |
| adam_tt | 1.597 |

Zhizi vs perturb (wave): Wilcoxon p ≈ 6×10⁻⁷.

![Synthetic method bars](docs/figures/synth_full_compare_bars.png)

*Figure: mean Vp RMSE across init / wave refine / TT solvers (`synth_full_compare_bars.png`).*

![Wave RMSE delta histogram](docs/figures/synth_wave_delta_hist.png)

*Figure: paired `zhizi_wave − perturb_wave` (negative = Zhizi better) (`synth_wave_delta_hist.png`).*

### Ray paths

![Example ray paths](docs/figures/example_paths.png)

*Figure: direct P/S rays for True / Zhizi init / Zhizi+wave models (`example_paths.png`).*

### Intermediate variables (ρ, envelope, picks, macro)

![Latent panel](docs/figures/latent_case_00.png)

*Figure: Z waveform, latent **ρ(t)**, wavefield envelope, and P/S pick curves with ground-truth onsets. ρ rises with strong energy (esp. S), aligned with phase arrivals (`latent_case_00.png`).*

![ρ vs distance](docs/figures/rho_vs_distance.png)

*Figure: mean ρ vs epicentral distance on latent sample cases (`rho_vs_distance.png`).*

![Macro / latent diagnostics](docs/figures/macro_latent_diagnostics.png)

*Figure: macro-implied Vp scale / contrast / Vs·Vp, ρ vs geometry, kernel soft prior, and scale–contrast coupling on STEAD (`macro_latent_diagnostics.png`).*

| Quantity | Reading from the plots |
|----------|-------------------------|
| `rho(t)` | Soft latent weight; spikes with energetic / S intervals |
| Envelope | Complex wavefield energy tracking phase structure |
| kernel_vp / vs | Dimensionless soft scales conditioning the head |
| macro (scale, contrast, ratio) | Low-dim deformation of the reference layered model |

Reproduce helper: `bash scripts/reproduce_macro_route.sh`.

---

## 7. Interpretability suite

Quantitative + visual evidence that internal variables align with physical phase structure (not post-hoc labels).

```bash
python run_interpret_suite.py --device cuda --copy-to-docs
# → outputs/interpret_suite/interpret_report.json
# → docs/figures/interpret/
```

### A. Kernel physics (Huygens vs Fresnel)

![Fresnel obliquity and kernel difference](docs/figures/interpret/kernel_obliquity_diff.png)

*Figure: Fresnel obliquity χ (left), log|K| Huygens (center), |K_Fresnel|−|K_Huygens| (right). Obliquity damps off-axis lags; kernel difference concentrates on longer causal lags.*

![Kernel row slice](docs/figures/interpret/kernel_row_slice.png)

*Figure: χ and |K| along one causal receiver row — forward cone structure.*

### B. Picking explainability (run20)

![Kernel contribution at GT P](docs/figures/interpret/kernel_contrib/kernel_contrib_00.png)

*Figure: Z trace, ρ(t), P/S envelopes, **|K| row at GT P index** (causal contributions), and pick curves. Kernel energy peaks near the P onset window.*

![ρ S-window vs noise](docs/figures/interpret/kernel_contrib/rho_s_over_noise_hist.png)

*Figure: ratio of mean ρ in S window vs pre-event noise; values > 1 indicate ρ tracks energetic phases.*

### C. Bridge latents (macro head)

![Bridge latent panel](docs/figures/interpret/bridge_latent/bridge_latent_00.png)

*Figure: frozen run20 features through the Zhizi bridge — ρ(t), envelope, P/S logits vs GT onsets.*

![Bridge ρ vs distance](docs/figures/interpret/bridge_latent/bridge_rho_vs_distance.png)

### D. Init → wave refine (Route A2)

![Inversion init vs refine](docs/figures/interpret/inversion_init_refine.png)

*Left: one-shot init VpRMSE vs after waveform refine (points below diagonal = refine helps). Right: paired zhizi−perturb wave delta (negative = Zhizi better).*

| Quantity | How to read it |
|----------|----------------|
| `rho(t)` | Soft latent weight; rises with S / high-energy intervals — **not** crustal density |
| χ obliquity | Fresnel aperture; forward lags weighted more than grazing paths |
| Kernel row | Which past samples causally contribute to a pick index |
| macro (scale, contrast, ratio) | Low-dim deformation of the reference layered model |

### E. Principle ablation: Huygens–Fresnel (completed)

`python run_huygens_fresnel_iterate.py` replayed picking + macro inversion with `--principle huygens_fresnel`.

![Picking principle compare](docs/figures/interpret/picking_principle_compare.png)

| Task | Huygens (run20) | Fresnel | Verdict |
|------|-----------------|---------|---------|
| Picking det F1 | 0.994 | **0.996** | Fresnel +0.002 |
| Picking P F1 | **0.959** | 0.925 | Fresnel −0.034 |
| Picking S F1 | **0.949** | 0.928 | Fresnel −0.022 |
| Route A2 win-rate | **93.8%** | 90.6% | still PASS |
| STEAD refine win-rate | **77.1%** | 77.1% | tie |

**Conclusion:** Fresnel does **not** replace the frozen run20 backbone (P/S regression). It remains an optional `--principle` for ablation; production path stays **run20 + macro**.

---

## 8. Repository layout

```
HNF/
├── hnf/                         # kernel, layers, field, picking, inversion, Zhizi bridge
│   ├── kernel.py density.py layers.py fmm.py field.py ...
│   ├── picking_model.py noise_cancel.py multiscale.py
│   ├── inversion_1d.py inversion_baselines.py acoustic_fwi_1d.py ray_paths.py
│   ├── zhizi_*.py
│   └── tests/
├── docs/figures/                # figures embedded in this README
│   └── interpret/               # interpretability suite mirrors
├── train_stead_picking.py
├── run11 … run20_stead_picking.py
├── train_zhizi_inversion.py
├── run_route_a2_waveform.py / run_zhizi_inv05_real.py / run_proof_suite.py
├── run_interpret_suite.py / run_huygens_fresnel_iterate.py
├── run_inv*.py
├── example_2d_reconstruction.py / train_stead.py
├── explain_stead_picking.py
├── scripts/reproduce_macro_route.sh
└── README_ZHIZI_INVERSION.md
```

---

## 9. Short reproduce path

```bash
# Picking (skip if run20 checkpoint exists)
python run20_stead_picking.py

# Macro head (skip if best_physics_head.pt exists)
python train_zhizi_inversion.py --head-mode macro --epochs 8 ...

# Performance proof (metrics + figures)
python run_proof_suite.py --device cuda --max-events 48 --n-synth 32

# Interpretability proof (kernel χ, contrib, ablations → docs/figures/interpret/)
python run_interpret_suite.py --device cuda --copy-to-docs
```

Open `outputs/proof_suite/proof_report.json`, `outputs/interpret_suite/interpret_report.json`, and `docs/figures/`.
