# Paper Roadmap: Evidence Status

This document tracks the eight research directions and the paper figure
checklist against concrete, paper-scale runs.

## Scale of current paper-grade runs

| Study | N | Script / output |
|------|---|-----------------|
| SNR robustness (Fig5) | 512 pick + 128 inv × 5 SNR × 2 modes | `outputs/paper_snr_robustness/` |
| Scene clustering + noise mining | 400 (380 after robust trim) | `outputs/paper_scene_clustering/` |
| Cluster-conditioned rediscovery | 380 × 35 candidates | `outputs/paper_cluster_rediscovery/` |
| Absolute-geo rediscovery + confirm | 380 + network/lon sensitivity | `outputs/paper_geo_rediscovery/`, `outputs/paper_geo_confirm/` |
| Cross-head transfer (priority laws) | 200 events × 4 heads | `outputs/paper_cross_head_transfer/` |
| Ambon cross-region TT | 64 catalog events | `outputs/paper_ambon_cross_region/` |
| OBS picking vs EQT/PhaseNet | 400 traces (chunk 201805) | `outputs/paper_obs_picking_compare/` |
| STEAD in-domain baselines | 2000 evt + 500 noise | `outputs/paper_stead_baseline_compare/` |
| Fig4 method board | packaged | `docs/figures/fig4_method_comparison.png` |
| Macro Q head (short train) | synthetic short run | `outputs/zhizi_inversion_macro_q/` |
| rho(t) vs classical attributes | 300 + 12 case panels | `outputs/paper_rho_attributes/` |
| Fig1 overview | concept figure | `docs/figures/fig1_hnf_overview.png` |

## Direction status

### 1. Scene clustering (conditional laws)
**Done (full rediscovery on clusters).**

Earlier clustering only screened 4 hand-picked relations. Full rediscovery
(`run_paper_cluster_rediscovery.py`, seed=11, same robust trim) screens
**35** pairwise/partial candidates globally and inside eligible clusters
(C2 n=4 skipped; min n=30). Support rule: bootstrap CI excludes 0 **and**
FDR q≤0.10.

Summary labels: **26 global / 6 scene-specific / 3 rejected**.

Priority (causal-chain / QC) global laws:

- `rho_p_lag -> init_tt` (partial ≈ −0.41)
- `rho_mean -> vp_mean` (partial ≈ −0.29)
- `noise_ratio -> pick_err_p` (partial ≈ +0.17)

Scene-specific (not global under partial controls):

- `noise_ratio -> init_tt` (partial): C1 ≈ +0.14, C3 ≈ +0.37
  (pairwise global sign is negative; partial flips → Simpson-style warning)
- `rho_mean -> init_tt` (partial): C1/C3 positive
- `rho_s_lag -> init_tt` (partial): C0/C1
- `rho_p_lag -> vp_mean` (partial): C3 only
- `s_prob_lag -> init_tt` (pairwise): C0/C1/C3

Treat strong geometry/head couplings (e.g. `rho_mean -> vpvs_mean`) as
**family-dependent** until cross-head transfer (below) is considered.

Figure: `docs/figures/cluster_rediscovery_summary.png`.

### 1b. Cross-head transfer of priority laws
**Done (n=200 × 4 heads).**

Script: `run_paper_cross_head_transfer.py` → `outputs/paper_cross_head_transfer/`.
Heads: `bridge_macro`, `bridge_residual`, `mixed_geo`, `stead_macro`.

| Relation | Transfer label | Notes |
|----------|----------------|-------|
| `rho_p_lag -> init_tt` | **head_robust** | ≈ −0.42 to −0.47 on all 4 heads |
| `rho_mean -> vp_mean` | **sign_unstable** | +0.89 (macro) vs −0.30 (mixed_geo) vs −0.80 (stead) |
| `rho_mean -> vpvs_mean` | head_robust* | same sign on 3/4; residual Vp/Vs collapses (constant) |
| `noise_ratio -> pick_err_p` | **head_independent_supported** | n=380 full CSV; not head-dependent |
| `noise_ratio -> init_tt` | head_specific_or_weak | unsupported on this all-sample slice |

\*Caveat: `bridge_residual` emits constant Vp/Vs on this slice → not a fair
counterexample for `rho→vpvs`, but `rho→vp` sign flips are decisive.

Paper claim update: keep `rho_p_lag→init_tt` and `noise_ratio→pick_err_p`;
downgrade `rho_mean→vp/vpvs` to head-family descriptive couplings.

Figure: `docs/figures/cross_head_transfer_summary.png`.

### 1c. Absolute geography (lat/lon) confirmation
**Done (CPU; n=380).**

Scripts: `run_paper_geo_rediscovery.py`, `run_paper_geo_confirm.py`.
Sample is **ZQ-dominated** (310/380); geo-kmeans C3≈309.

| Claim | Label | Notes |
|-------|-------|-------|
| `noise_ratio → pick_err_p` | **geo-confirmed** | survives lat/lon and `is_ZQ`; holds in ZQ-only |
| `rho_p_lag → init_tt` | **geo-confirmed** | same |
| `rho_mean → vp_mean` | geo-survives / head-unstable | keep secondary |
| absolute `source_lat → pick_err_p` | **reinterpreted** | collapses after `is_ZQ` (network/region proxy) |
| within-ZQ `source_lon → pick_err_p` | local geo | ρ≈0.15 inside ZQ; not universal lat physics |

Figures: `docs/figures/geo_*.png`. Details in `docs/KNOWLEDGE_MINING.md`.

### 2. Noise-branch utility
**Done (mining enabled).**

With noise-cancel ON, `noise_ratio = energy(n_sim)/(energy(n_sim)+energy(u_denoised))`
is a useful QC latent:

- predicts higher P pick error globally
- in some clusters also couples to init TT misfit

Denoise also helps mid-SNR picking (see Fig5): at 10 dB, denoise-on P-F1
0.756 vs bypass 0.712.

### 3. Density ρ vs Vp/Vs/Q
**Clarified + Q stub done; physical density deferred.**

- `rho(t)` remains a soft latent, not crustal density
- Physics head now supports optional `predict_q=True` (macro Q scale around 120)
- Physical density still deferred: travel-time weak constraint / non-uniqueness

### 4. Robustness (SNR + generalization)
**SNR done; Ambon cross-region TT done; picking cross-dataset still open.**

Fig5 now has two complementary halves:

1. **STEAD SNR** (`docs/figures/fig5_snr_robustness.png`) — picking robustness
2. **Ambon Indonesia geometry/TT** (`docs/figures/fig5_ambon_cross_region.png`) —
   cross-region travel-time inversion on real BMKG/ITB catalog geometry

Ambon data: `external_data/ambon_mendeley/` (catalog + VELEST model; **no waveforms**).
Script: `run_paper_ambon_cross_region.py` → `outputs/paper_ambon_cross_region/`
(n=64 events, success = Vp RMSE≤3 and TT misfit≤1).

| Method | Success | Median Vp RMSE (successes) |
|--------|--------:|---------------------------:|
| Gauss-Newton (damped) | 64/64 (100%) | 0.60 |
| Homogeneous grid | 64/64 (100%) | 1.60 |
| L-BFGS | 63/64 (98%) | 0.68 |
| HNF-Adam | 32/64 (50%) | **0.51** |

Interpretation: on Ambon geometry, classical damped GN is the most stable
transfer baseline; when HNF-Adam converges it can be more accurate, but it
diverges on half the events. Report success rate, not raw mean RMSE.

**STEAD in-domain is real for all three models** (same protocol, 2000 events +
500 noise subset; HNF also has full-test `test_metrics.json`):

| Model | det_f1 | P-F1 | S-F1 |
|------|-------:|-----:|-----:|
| HNF(run28-50ep) | **0.998** | **0.980** | **0.965** |
| HNF(run20) full test | 0.994 | 0.959 | 0.949 |
| HNF(run20) subset | 0.996 | 0.962 | 0.954 |
| EQT(STEAD) subset | **0.999** | **0.989** | **0.971** |
| PhaseNet(STEAD) subset | 0.997 | 0.949 | 0.959 |

Script/figure: `run_paper_stead_baseline_compare.py`,
`docs/figures/fig5_stead_baseline_compare.png`.
OBS low scores do **not** contradict this — they are cross-domain
(land STEAD → ocean OBS).

Picking zero-shot on SeisBench OBS (`run_paper_obs_picking_compare.py`, chunk
`201805`, n=400, **protocol v3** = v2 + HNF logits→sigmoid):

| Model | role | P-F1 | S-F1 |
|------|------|-----:|-----:|
| HNF(run20/STEAD) | zero-shot | 0.086 | 0.212 |
| EQT(STEAD) | zero-shot baseline | **0.273** | 0.588 |
| PhaseNet(STEAD) | zero-shot baseline | 0.095 | **0.591** |
| EQT(OBS) | domain reference (4C) | 0.680 | 0.689 |
| PhaseNet(OBS) | domain reference (4C) | 0.656 | 0.658 |

Protocol fixes:
- PhaseNet labels are `PSN` (earlier wrongly decoded as `NPS`)
- STEAD models use `peak` norm; OBS models use `std` norm
- HNF uses per-channel demean+std (matches STEAD training)
- HNF forward returns **logits**; must `sigmoid` before threshold (matches
  `train_stead_picking.py`) — fixing this barely changes OBS F1
- Drop incomplete 3C traces (ZH / Z1H)

OBS failure mode for HNF: det head often NaN; P argmax median error ~10–14 s
(domain shift), while S is relatively better. Fair comparison is the
STEAD-trained trio; OBS-pretrained models are upper-bound references.

Figure: `docs/figures/fig5_obs_picking_compare.png`.

STEAD SNR table:

| SNR (dB) | P-F1 (denoise) | S-F1 (denoise) | P-F1 (bypass) |
|----------|----------------|----------------|---------------|
| 20 | 0.931 | 0.934 | 0.919 |
| 15 | 0.842 | 0.860 | 0.827 |
| 10 | 0.756 | 0.778 | 0.712 |
| 5 | 0.564 | 0.714 | 0.527 |
| 0 | 0.114 | 0.759 | 0.138 |

### 5. rho(t) vs classical attributes + case library
**Done.**

On n=300:

- window corr(rho, envelope) @P ≈ 0.70
- window corr(rho, STA/LTA) @P ≈ 0.70
- peak-lag Spearman(rho, env) @P ≈ 0.75 (CI [0.69, 0.82])

So `rho(t)` is **aligned with classical energy/onset attributes**, but remains
an internal Huygens-conditioned latent rather than a drop-in STA/LTA clone.
Case library: `outputs/paper_rho_attributes/cases/`.

### 6. Q + pseudo-2D confidence
**Q API + train flag + short synthetic run; imaging confidence exists.**

- `ZhiziPhysicsHead(..., predict_q=True)` emits Q scale
- `train_zhizi_inversion.py --predict-q` and bridge loader now persist/restore it
- short macro run: `outputs/zhizi_inversion_macro_q/`
  (best val Vp RMSE ≈ 0.34; `out` dim=4 confirms Q channel; Q RMSE=0 because
  current synthetic TT loss does not supervise amplitude/Q — needs amplitude-aware
  follow-up)
- Phase E/F coverage / uncertainty / trust masks already exist
  (`docs/figures/phase_ef_overview.png`)

### 7. Method comparison section
**Packaged.**

Fig4 board: `docs/figures/fig4_method_comparison.png`
(`run_paper_fig4_board.py` from inv_full_compare + proof_suite assets).

### 8. Paper figure checklist

| Figure | Status | Asset |
|------|--------|-------|
| Fig1 concept + formula + flowchart | **EXISTS** | `docs/figures/fig1_hnf_overview.png` |
| Fig2 γ/ω kernel behavior | **EXISTS** | `docs/figures/interpret/kernel_gamma_omega_semantics.png` |
| Fig3 picking sweep + panels | **EXISTS** | threshold sweep + kernel_contrib / latent panels |
| Fig4 inversion improvement | **EXISTS** | `docs/figures/fig4_method_comparison.png` |
| Fig5 SNR / generalization | **PARTIAL+** | SNR + Ambon TT + OBS picking compare (pilot chunk) |
| Fig6 pseudo-2D + confidence | **EXISTS** | `phase_ef_overview.png` + Phase F trust masks |
| Fig7 causal chain + counterfactual | **EXISTS** | interpret causal / counterfactual panels |

## Reproduce commands

```bash
python run_paper_fig1_overview.py
python run_paper_snr_robustness.py --max-pick-events 512 --max-inv-events 128
python run_paper_scene_clustering.py --max-events 400 --n-clusters 4
python run_paper_cluster_rediscovery.py --seed 11
python run_paper_cross_head_transfer.py --max-events 200
python run_paper_ambon_cross_region.py --n-events 64 --steps 400
python run_paper_obs_picking_compare.py --max-events 400
python run_paper_fig4_board.py
python run_paper_rho_vs_attributes.py --max-events 300 --n-cases 12
python train_zhizi_inversion.py --dataset synthetic --head-mode macro --predict-q \
  --n-train 80 --n-val 16 --epochs 12 --output-dir outputs/zhizi_inversion_macro_q
```

## Future application domains (design)

| Domain | Doc | Priority vs current work |
|--------|-----|--------------------------|
| II — AD/FTD EEG | `hnf/eeg_*`, `train_eeg.py` | After STEAD/OBS bandwidth frees |
| III — Sparse 4D Flow → constitutive discovery | [`DOMAIN_III_FLUID_RHEOLOGY.md`](DOMAIN_III_FLUID_RHEOLOGY.md) | Design only; after EEG Stage-1 |

## Next minimum paper upgrades

1. Improve OBS transfer: more OBS chunks, HNF light-adapt / threshold sweep, PhaseNet post-processing
2. Stabilize HNF-Adam on Ambon (step size / damping / better init) or report it as conditional
3. Longer amplitude-aware Q training / evaluation
