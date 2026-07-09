# HNF Knowledge Mining Report

- n rows: 72
- sample table: `sample_level_stats.csv`
- overview: `knowledge_overview.png`

## Screened relations
- `distance_km -> rho_mean`: spearman=0.003, 95% CI=[-0.250, 0.236], p≈0.982, q≈1
- `source_depth_km -> rho_mean`: spearman=-0.018, 95% CI=[-0.298, 0.229], p≈0.881, q≈1
- `rho_mean -> vp_mean`: spearman=0.085, 95% CI=[-0.158, 0.319], p≈0.479, q≈1
- `rho_mean -> vpvs_mean`: spearman=0.087, 95% CI=[-0.153, 0.308], p≈0.469, q≈1
- `kernel_vp -> vp_mean`: spearman=nan, 95% CI=[nan, nan], p≈nan, q≈1
- `kernel_vs -> vs_mean`: spearman=nan, 95% CI=[nan, nan], p≈nan, q≈1
- `p_prob_lag -> refined_tt`: spearman=0.068, 95% CI=[-0.167, 0.326], p≈0.575, q≈1
- `s_prob_lag -> refined_tt`: spearman=0.126, 95% CI=[-0.112, 0.374], p≈0.295, q≈1
- `pick_err_p -> refined_tt`: spearman=0.033, 95% CI=[-0.222, 0.293], p≈0.784, q≈1
- `pick_err_s -> refined_tt`: spearman=0.162, 95% CI=[-0.067, 0.402], p≈0.177, q≈1