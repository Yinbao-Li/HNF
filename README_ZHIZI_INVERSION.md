# Physics Decoder (Zhizi / macro head)

Maps frozen HNF picking latents → layered `vp/vs` (FWI-lite **initializer**).

**Current preferred stack (2026-07):**

| Piece | Path |
|-------|------|
| Picking | `outputs/run28/28_ms_fresnel_phys_20ep/best.pt` |
| Decoder | `outputs/physics_decoder_run28_macro/best_physics_head.pt` |
| Large-N A2 | `outputs/route_a2_run28_macro_n256/` (**judge on init**) |
| Legacy run20-macro | `outputs/zhizi_inversion_bridge_macro/` (strong **wave**-win, weak init) |

See master write-up: [`README.md`](README.md). Plan: [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

## Results (large-N)

| Setting | Outcome |
|---------|---------|
| run28 macro val | Vp RMSE ≈ **0.136** |
| A2 n=256 init | Zhizi **0.173** vs perturb 0.146 (init-win **41%**); run20-macro init 0.304 |
| A2 n=256 wave-win | run28 ~**53%**; run20-macro **91%** — do not overclaim wave-win for run28 |
| STEAD refine n=500 | win **69.6%** (`proof_suite_run28_n500`) |

## Train (run28)

```bash
python tools/train_zhizi_inversion.py \
  --checkpoint outputs/run28/28_ms_fresnel_phys_20ep/best.pt \
  --head-mode macro --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/physics_decoder_run28_macro
```

Optional: `--kernel-summary --mid-tt-weight 0.08` → `physics_decoder_run28_macro_ks`.

## Eval

```bash
python scripts/inversion/run_route_a2_waveform.py \
  --checkpoint outputs/run28/28_ms_fresnel_phys_20ep/best.pt \
  --physics-head outputs/physics_decoder_run28_macro/best_physics_head.pt \
  --head-mode macro --n-test 256 --fwi-steps 60 --device cuda \
  --output-dir outputs/route_a2_run28_macro_n256
```

Code: `hnf/physics_decoder.py` (shim `zhizi_inversion_bridge.py`).
