# Zhizi Inversion Bridge (macro head)

Frozen run20 picking backbone + macro Physics Head (`scale` / `contrast` / `Vs ratio`) → `vp0/vs0`, then short waveform refinement. Goal: a **better FWI-lite initializer**.

See the full HNF write-up and embedded figures in [`README.md`](README.md).

## Results

| Experiment | Outcome |
|------------|---------|
| Route A2, 32 events | Win-rate **≈0.94**; VpRMSE Zhizi **0.924** vs perturb **0.982** |
| Route A2, 64 events | Win-rate **≈0.875**; **0.935** vs **0.977** |
| STEAD geom-aware refine, 48 events | Win-rate **77.1%**; TT mean **3.08** vs **11.22** |
| Macro short train | Best Val Vp RMSE **≈0.277** (~epoch 4) |

## Assets

- Picking: `outputs/run20/20_wrongpeak_sharp/best.pt`
- Physics head: `outputs/zhizi_inversion_bridge_macro/best_physics_head.pt`
- Mode: `--head-mode macro`

## Reproduce

```bash
python train_zhizi_inversion.py \
  --head-mode macro --epochs 8 --n-train 96 --n-val 16 \
  --unrolled-weight 0.5 --unrolled-steps 5 \
  --vp-sup-weight 0.05 --lr 3e-3 \
  --output-dir outputs/zhizi_inversion_bridge_macro

python run_route_a2_waveform.py \
  --head-mode macro \
  --physics-head outputs/zhizi_inversion_bridge_macro/best_physics_head.pt \
  --n-test 32 --fwi-steps 60 --device cuda

python run_proof_suite.py --device cuda --max-events 48 --n-synth 32 \
  --output-dir outputs/proof_suite

bash scripts/reproduce_macro_route.sh
```

## Pipeline

```
waveform → frozen Zhizi features (rho / envelope / kernel / picks)
        → macro head → vp0/vs0
        → short waveform / TT refine → m*
```

## Figures

| Path | Content |
|------|---------|
| `docs/figures/training_curves.png` | Macro-head training |
| `docs/figures/stead_refine_scatter.png` | STEAD refine scatter |
| `docs/figures/synth_full_compare_bars.png` | Synthetic baselines |
| `docs/figures/example_paths.png` | Ray paths |
| `docs/figures/latent_case_00.png` | ρ(t) / envelope / picks |
| `docs/figures/macro_latent_diagnostics.png` | Macro & latent diagnostics |

Interpretability (kernel χ, contrib rows, Fresnel ablation): `python run_interpret_suite.py --device cuda --copy-to-docs` → `docs/figures/interpret/`.
