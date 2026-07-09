# HNF Interpretability Report

## Kernel physics
- Mean |K_Fresnel − K_Huygens|: 0.7620272040367126
- See `kernel_obliquity_diff.png`, `kernel_row_slice.png`

## Picking (run20 vs Fresnel)
- Delta: {'det_f1': 0.0020927960090827424, 'p_f1': -0.033952444950457905, 's_f1': -0.021503411065077693}

## Kernel gamma / omega semantics
- See `kernel_gamma_omega_semantics.png`
- gamma range: [0.09539812803268433, 3.3683700561523438]
- omega range: [0.9296556711196899, 5.025640964508057]
- wave_speed range: [4.51204776763916, 8.001336097717285]

## Latent rho
- S-window / noise rho ratio cases: 4

## Counterfactual response
- See `counterfactual_response_panel.png`

## Temporal lag statistics
- n cases: 24

## Branch parameter ablation
- See `branch_parameter_ablation.png`
- weak bridge propagation: True
- max vp span: 0.0
- max vs span: 0.0

## Summary panel
- See `interpretability_summary_panel.png`

## Causal chain
- See `causal_chain_graph.png`, `causal_wave_summary.png`

## Joint latent / physical summary
- n cases: 24

## Vp/Vs sensitivity
- See `vp_vs_tt_sensitivity.png`

## Inversion
- {'run20': {'mean_zhizi_wave': 0.9244259316474199, 'mean_perturb_wave': 0.982438649982214, 'win_frac': 0.9375}, 'fresnel': {'mean_zhizi_wave': 0.9368581119924784, 'win_frac': 0.90625}}

Run: `python run_interpret_suite.py --device cuda --copy-to-docs`