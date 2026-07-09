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

## Robust subset
- kept: 63 / 72
- trim_quantile: 0.950
- tt_cut: 13.888
- pick_err_p_cut: 0.619
- pick_err_s_cut: 0.450

## Robust partial relations
- `rho_mean -> vp_mean | distance_km, source_depth_km`: partial=0.227, 95% CI=[0.010, 0.447], p≈0.0792, q≈0.229
- `rho_mean -> vpvs_mean | distance_km, source_depth_km`: partial=0.204, 95% CI=[-0.051, 0.401], p≈0.115, q≈0.229
- `p_prob_lag -> refined_tt | distance_km, source_depth_km, pick_err_p`: partial=0.119, 95% CI=[-0.162, 0.356], p≈0.355, q≈0.37
- `s_prob_lag -> refined_tt | distance_km, source_depth_km, pick_err_s`: partial=0.116, 95% CI=[-0.166, 0.363], p≈0.37, q≈0.37
- `rho_p_lag -> refined_tt | distance_km, source_depth_km, pick_err_p`: partial=0.248, 95% CI=[-0.015, 0.483], p≈0.0543, q≈0.229
- `rho_s_lag -> refined_tt | distance_km, source_depth_km, pick_err_s`: partial=0.146, 95% CI=[-0.106, 0.395], p≈0.259, q≈0.37

## Mediation-style screen
- `rho_mean -> p_prob_lag -> refined_tt`: rho_xm=0.107, rho_my=0.068, rho_xy=-0.029, chain_score=0.007
- `rho_mean -> s_prob_lag -> refined_tt`: rho_xm=0.306, rho_my=0.126, rho_xy=-0.029, chain_score=0.039
- `rho_p_lag -> p_prob_lag -> refined_tt`: rho_xm=0.580, rho_my=0.068, rho_xy=0.058, chain_score=0.039
- `rho_s_lag -> s_prob_lag -> refined_tt`: rho_xm=0.381, rho_my=0.126, rho_xy=0.040, chain_score=0.048

## Bucket outputs
- overview: `knowledge_bucket_panels.png`
- distance buckets: 4
- depth buckets: 4

## Head compare
- `zhizi_inversion_bridge_macro`: vp_mean_avg=5.650, vs_mean_avg=2.875, vpvs_mean_avg=1.967, init_tt_avg=51.167±60.357
- `zhizi_inversion_bridge_residual`: vp_mean_avg=5.300, vs_mean_avg=3.060, vpvs_mean_avg=1.733, init_tt_avg=64.411±64.801
- `zhizi_inversion_mixed_geo`: vp_mean_avg=5.150, vs_mean_avg=2.669, vpvs_mean_avg=1.931, init_tt_avg=38.904±55.051
- `zhizi_inversion_stead_macro`: vp_mean_avg=4.261, vs_mean_avg=2.121, vpvs_mean_avg=2.011, init_tt_avg=96.893±163.435