# Experiment plan (post-run28)

Documentation frame = README Parts I–IV.
**Step 5–6 DONE. Step 7 Fluid Stage-0/1 + RACLETTE Stage-0b DONE.**

Primary ckpt: `outputs/run28/28_ms_fresnel_phys_20ep/best.pt`
Preferred Decoder: `outputs/physics_decoder_run28_macro/`
Outputs index: [`../outputs/CURRENT.md`](../outputs/CURRENT.md),
[`../outputs/SUMMARY.json`](../outputs/SUMMARY.json).

| Step | Status |
|------|--------|
| 0 Promote run28 metrics | DONE |
| 1 README Part I metrics | DONE (refreshed 2026-07-15) |
| 2 Interpret + probing | DONE first pass; fold figures ongoing |
| 3 Decoder upgrade + large-N | DONE; **claim = init** |
| **4 OBS multi-chunk** | **DONE** + **P–S tradeoff grid** (selected HNF trunk-tail/L1200: P=0.374 S=0.649) |
| **5 Mining / reparam** | **DONE** mining v1–v4 + `outputs/reparam_suite_run28/` |
| 6 EEG | **DONE** Stage-1 + baselines (`adftd_hnf_stage1/`, `adftd_baseline_compare/`) |
| 7 Fluid | Stage-0/1 **DONE**; RACLETTE Stage-0b **DONE** (inside-vessel vel_rel≈0.79 @10% — hard) |
| **8 Foveated** | **DONE (first board)** — test P=0.917 / S=0.940 @7.44 gazes vs dense 0.954/0.955 |

### OBS freeze note (2026-07-16)

Step 4 results stand as the final OBS board under random `p_offset`.  
**OBS full retrain** (`run28_obs_full_800`, holdout P=0.303 S=0.711) is the canonical HNF OBS checkpoint.  
Do **not** schedule further OBS structure / noise-branch / adapt experiments.  
Next bandwidth → **Step 8 Foveated** (`scripts/experiments/run_foveated_stage1.py`).

### Step 7c RACLETTE Stage-0b (2026-07-16) — DONE

Preprocessed 900 GT in-plane slices (32×32) from VirtualSubject_n001 via pyvista
(`/usr/bin/python3`; anaconda 3.8 cannot install `pyvista-zstd`).

| Metric | Test |
|--------|-----:|
| vel_rel (full field) | 1.004 |
| **vel_rel (inside vessel)** | **0.793** |

**Takeaway:** RACLETTE anatomy is much harder than synthetic Poiseuille/Couette
at the same 10% keep; first pass does **not** meet a strong reconstruction bar.
Cache: `external_data/raclette_cache/gt_slices.npz`.

```bash
/usr/bin/python3 tools/preprocess_raclette_slices.py
PYTHONPATH=. python scripts/experiments/run_fluid_stage0b_raclette.py --skip-preprocess --epochs 40
```

### Step 7b Fluid Stage-1 (2026-07-16) — DONE

Newtonian vs Carreau ID + θ recovery from 10% sparse 2D channels. 50ep ~16 min.

| Metric | Test |
|--------|-----:|
| **family_acc** | **0.799** (N 0.780 / C 0.818) |
| η / η₀ rel. err | **0.267** |
| θ rel. err (masked) | 0.311 |
| vel_rel | 0.109 |

vs Stage-0 η_rel 0.59 → **improved**. Still above the aspirational &lt;10% η target.
Confusion mostly balanced; not a RACLETTE claim.

```bash
PYTHONPATH=. python scripts/experiments/run_fluid_stage1.py --epochs 50 --device cuda
```

Artifacts: `outputs/fluid/stage1_constitutive/`,
`hnf/fluid_constitutive.py`, `hnf/fluid_constitutive_model.py`.

### Step 7 Fluid Stage-0 (2026-07-16) — DONE

Synthetic 2D sparse→dense (keep=10%) + η head. 40ep ~6 min.

| Metric | Value |
|--------|------:|
| test vel_rel (overall) | **0.330** |
| poiseuille / couette / vortex | 0.085 / 0.052 / 0.869 |
| η rel. err | 0.591 (weak — Stage-1 target) |

RACLETTE `.pv` needs pyvista — deferred; volumes on disk.
Next: Stage-1 constitutive synthetic (Newtonian / Carreau ID + η recovery).

```bash
PYTHONPATH=. python scripts/experiments/run_fluid_stage0.py --epochs 40 --device cuda
```

Artifacts: `outputs/fluid/stage0_synth/`, modules `hnf/fluid_{synth,dataset,model}.py`.

### Step 6 EEG Stage-1 (2026-07-16) — DONE

50ep ~73 min; best val AUC **0.796** (ep29). Held-out test (non-overlap, 18 subjects):

| Metric | Value |
|--------|------:|
| epoch accuracy | 0.675 |
| macro-F1 | 0.647 |
| **subject accuracy** | **0.778** |
| **macro-AUC** | **0.841** |

Artifacts: `outputs/eeg/adftd_hnf_stage1/{best.pt,test_metrics.json}`, figures `docs/figures/eeg/`.

### Step 6b EEG baselines (2026-07-16) — DONE

Same subject split / 19×1280 / 50ep / best-val-AUC selection:

| Model | subject_acc | macro-AUC | epoch_acc | macro-F1 | params |
|-------|------------:|----------:|----------:|---------:|-------:|
| **HNF(stage1)** | **0.778** | **0.841** | 0.675 | **0.647** | 89k |
| EEGNet | 0.722 | 0.818 | **0.695** | 0.613 | 3.3k |
| Shallow1D | 0.500 | 0.840 | 0.565 | 0.459 | 125k |

Board: `outputs/eeg/adftd_baseline_compare/compare_summary.{json,md}`

```bash
PYTHONPATH=. python scripts/experiments/run_eeg_baseline_compare.py --epochs 50 --device cuda
```

```bash
PYTHONPATH=. python scripts/experiments/run_eeg_stage1.py --epochs 50 --device cuda
# or:
PYTHONPATH=. python tools/train_eeg.py --no-synthetic --multi-scale --principle huygens_fresnel \
  --output-dir outputs/eeg/adftd_hnf_stage1
PYTHONPATH=. python tools/eval_eeg.py --no-synthetic --checkpoint outputs/eeg/adftd_hnf_stage1/best.pt
```

### Step 5 reparam (2026-07-16)

Script: `scripts/interpret/run_reparam_suite.py`

| Track | Artifact | Takeaway |
|-------|----------|----------|
| Analytic medium | `analytic_medium_distance_fits.png` | ρ/vp/vpvs vs distance poly fits (event-wise) |
| Classical residual | `velocity_residual_vs_classical.png` | Decoder layered vp/vs vs AK135 (Ambon table) |
| Low-rank K | `kernel_svd_lowrank.png` | SVD spectrum + ranks 1/2/4/8 recon error |

```bash
PYTHONPATH=. python scripts/interpret/run_reparam_suite.py \
  --checkpoint outputs/run28/28_ms_fresnel_phys_20ep/best.pt \
  --physics-head outputs/physics_decoder_run28_macro/best_physics_head.pt \
  --compare ak135 --svd-ranks 1,2,4,8 \
  --output-dir outputs/reparam_suite_run28
```

### Step 3 evidence (keep)

- A2 n=256 init: run28 macro **0.173** vs run20 macro 0.304 vs perturb 0.146
- A2 wave-win: run28 ~53–56% vs run20-macro **91%** → do not overclaim wave-win
- Proof n=500 STEAD refine win **69.6%**

### Step 4 checklist

- [x] Multi-chunk OBS — **201805–201808**
- [x] Fairness: same eval domain + same treatment; disjoint holdout
- [x] Kill absolute-time cue: **p_offset ~ U(4,12)** per event
- [x] Matched adapt: HNF heads / heads+onset vs EQT heads

### Fairness rule

All numbers **eval on OBS**. Do not mix ZS vs adapt in one claim.
Canonical split: `outputs/obs_matched_adapt_split_randoffset/split.json`
(train 2400 / holdout 800, seed11, offset∈[4,12], **0 overlap**).

Fixed `p_offset=8` tables are **superseded** (EQT adapt P≈0.98 was a clock prior).

### A. Zero-shot (random-offset holdout)

| Model | P-F1 | S-F1 |
|------|-----:|-----:|
| HNF(run28/STEAD) | 0.201 | 0.453 |
| EQT(STEAD) | **0.543** | **0.660** |
| PhaseNet(STEAD) | 0.417 | 0.563 |

### B. Matched light-adapt (same holdout)

| Model | P-F1 | S-F1 | note |
|------|-----:|-----:|------|
| HNF heads-only @800 | 0.276 | 0.536 | |
| HNF heads+onset @800 | 0.303 | **0.702** | best **S** |
| HNF heads+onset @1600 | 0.354 | 0.601 | |
| **HNF trunk-tail @1600** | **0.370** | 0.609 | best **P** |
| **EQT(STEAD+OBS-adapt)** | **0.589** | **0.745** | |

Artifacts:
- `outputs/obs_light_adapt_run28_randoff_{heads,onset,onset1600,trunktail1600}/`
- `outputs/obs_light_adapt_eqt_randoff/`
- `outputs/obs_step4_randoffset_{zs,matched_adapt,pboost_*}/`

**Takeaway:** Random offset collapses EQT adapt P 0.98→**0.59**.  
HNF **S** peaks at **0.70** (heads+onset@800), near EQT adapt.  
Raising `seq_len=1600` + `trunk-tail` lifts HNF **P** 0.30→**0.37** but trades off some S.  
Naive **EQT-grid** `seq_len=6000` (100 Hz×60 s) is **not viable** for HNF: matched adapt holdout P/S ≈ **0.06 / 0.12**.  
Local 100 Hz refine was a possible P lever — **shelved** (no further OBS work).

```bash
# EQT-grid attempt (negative; for ablation only)
PYTHONPATH=. python tools/train_obs_light_adapt.py \
  --checkpoint outputs/run28/28_ms_fresnel_phys_20ep/best.pt \
  --split-json outputs/obs_matched_adapt_split_randoffset/split.json \
  --seq-len 6000 --tune trunk-tail --force-sparse-band \
  --cap-local-window-sec 2.0 --batch-size 1 \
  --output-dir outputs/obs_light_adapt_run28_randoff_eqtgrid6000
```

```bash
PYTHONPATH=. python tools/obs_matched_split.py \
  --output outputs/obs_matched_adapt_split_randoffset/split.json \
  --p-offset-min 4 --p-offset-max 12

PYTHONPATH=. python tools/train_obs_light_adapt.py \
  --checkpoint outputs/run28/28_ms_fresnel_phys_20ep/best.pt \
  --split-json outputs/obs_matched_adapt_split_randoffset/split.json \
  --tune trunk-tail --seq-len 1600 --det-loss-weight 0 \
  --p-loss-weight 2.2 --device cuda \
  --output-dir outputs/obs_light_adapt_run28_randoff_trunktail1600

PYTHONPATH=. python scripts/paper/run_paper_obs_picking_compare.py \
  --checkpoint outputs/obs_light_adapt_run28_randoff_trunktail1600/best.pt \
  --hnf-label 'HNF(adapt/trunk-tail/1600)' --seq-len 1600 \
  --eqt-adapt-checkpoint outputs/obs_light_adapt_eqt_randoff/best.pt \
  --split-json outputs/obs_matched_adapt_split_randoffset/split.json \
  --output-dir outputs/obs_step4_randoffset_pboost_trunktail
```

### Step 8 Foveated active perception (2026-07-16) — DONE (first board)

Dual-fovea engine on 60 s STEAD windows (`seq_len=6000`), frozen run28 via
`shift_downsample`.

| Piece | Location |
|-------|----------|
| Core | `hnf/foveated/` |
| Test board | `scripts/experiments/run_foveated_test_board.py` → `outputs/foveated/test_board/` |
| Gaze ablation | `tools/eval_foveated_gaze_ablation.py` → `outputs/foveated/gaze_ablation/` |
| Figures | `docs/figures/foveated_gaze_{trajectory_panel,ablation}.png` |

**STEAD test (n=800):** dense P/S **0.954 / 0.955**; foveated ZS **0.917 / 0.940** @ **7.44** gazes
(0.097 s/trace vs dense 0.006). Unlimited budget saturates ~7.5 gazes.

**OBS holdout ZS (n=800):** foveated STEAD **0.064 / 0.339** vs dense STEAD **0.201 / 0.453**
vs EQT **0.543 / 0.660** — foveated does **not** transfer; OBS-full-in-fovea still < dense OBS-full.
Board: `outputs/foveated/obs_zs_board/`.

```bash
PYTHONPATH=. python scripts/experiments/run_foveated_test_board.py --max-events 800 --device cuda
PYTHONPATH=. python scripts/experiments/run_foveated_obs_zs_board.py --device cuda
```

Optional next: OBS-aware peripheral scanner / light adapt; or native 100 Hz short-window fovea backbone.
