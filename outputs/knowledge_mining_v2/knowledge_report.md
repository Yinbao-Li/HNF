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

## Partial relations
- `rho_mean -> vp_mean | distance_km, source_depth_km`: partial=0.131, 95% CI=[-0.089, 0.346], p≈0.277, q≈0.462
- `rho_mean -> vpvs_mean | distance_km, source_depth_km`: partial=0.123, 95% CI=[-0.136, 0.324], p≈0.308, q≈0.462
- `p_prob_lag -> refined_tt | distance_km, source_depth_km, pick_err_p`: partial=0.144, 95% CI=[-0.073, 0.371], p≈0.231, q≈0.462
- `s_prob_lag -> refined_tt | distance_km, source_depth_km, pick_err_s`: partial=0.086, 95% CI=[-0.134, 0.334], p≈0.478, q≈0.573
- `rho_p_lag -> refined_tt | distance_km, source_depth_km, pick_err_p`: partial=0.203, 95% CI=[0.001, 0.420], p≈0.0922, q≈0.462
- `rho_s_lag -> refined_tt | distance_km, source_depth_km, pick_err_s`: partial=0.059, 95% CI=[-0.165, 0.303], p≈0.625, q≈0.625

## Mediation-style screen
- `rho_mean -> p_prob_lag -> refined_tt`: rho_xm=0.107, rho_my=0.068, rho_xy=-0.029, chain_score=0.007
- `rho_mean -> s_prob_lag -> refined_tt`: rho_xm=0.306, rho_my=0.126, rho_xy=-0.029, chain_score=0.039
- `rho_p_lag -> p_prob_lag -> refined_tt`: rho_xm=0.580, rho_my=0.068, rho_xy=0.058, chain_score=0.039
- `rho_s_lag -> s_prob_lag -> refined_tt`: rho_xm=0.381, rho_my=0.126, rho_xy=0.040, chain_score=0.048

## Bucket outputs
- overview: `knowledge_bucket_panels.png`
- distance buckets: 4
- depth buckets: 4