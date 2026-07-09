# Knowledge Mining

## Goal

The next stage after interpretability is **statistical knowledge mining**:

1. export one unified sample-level table
2. test candidate regularities with uncertainty
3. reject weak stories early
4. keep causal-chain consistency in every claim

This stage is explicitly about `rho`, `vp`, `vs`, `vp/vs`, `gamma`, `omega`,
geometry, pick quality, and travel-time fit quality.

## First-pass export

`run_knowledge_mining.py` now exports:

- `sample_level_stats.csv`
- `sample_level_stats.json`
- `knowledge_report.json`
- `knowledge_report.md`
- `knowledge_overview.png`

The current first-pass run used `72` STEAD test events and included:

- geometry: `distance_km`, `source_depth_km`
- quality: `pick_err_p`, `pick_err_s`, `init_tt`, `refined_tt`
- latent timing: `rho_mean`, `rho_peak`, `rho_p_lag`, `rho_s_lag`
- envelope / pick timing: `env_p_lag`, `env_s_lag`, `p_prob_lag`, `s_prob_lag`
- physical outputs: `vp_mean`, `vs_mean`, `vpvs_mean`, layerwise spread
- kernel / branch constants: `gamma_p0`, `omega_p0`, `gamma_s0`, `omega_s0`,
  `kernel_vp`, `kernel_vs`

## Current conclusion

The first screened relation set did **not** produce a robust event-level
statistical law after bootstrap CI and FDR correction.

This is important, because it avoids over-claiming. In particular:

- `distance_km -> rho_mean` is currently near zero
- `source_depth_km -> rho_mean` is currently near zero
- `rho_mean -> vp_mean` and `rho_mean -> vp/vs` are weak in the present sample
- `pick_err_p/s -> refined_tt` are not yet strong enough to be called stable

So the current state is:

> We have a working mining pipeline, but not yet a confirmed new law.

## Why `gamma/omega` need special treatment

In the current model, `gamma`, `omega`, `kernel_vp`, and `kernel_vs` are
**global learned branch parameters**, not per-event variables. Therefore:

- event-level correlation of `kernel_vp -> vp_mean` is not meaningful
- NaN correlation here is expected, because the predictor is effectively
  constant across the exported event table

This is consistent with the updated interpretability conclusion:

- branch-local `gamma/omega` perturbations clearly change kernel support and
  pick timing
- but they only weakly propagate to downstream `vp/vs` in the current local
  ablation

Therefore future mining for `gamma/omega` should use one of these routes:

1. **local sensitivity mining** from ablation curves
2. **cross-checkpoint mining** across differently trained models
3. **branch-to-latent mediation analysis** rather than naive event-level
   correlation

## Causal-chain requirement

Any future claim should be checked along a causal chain, not only with a
single correlation coefficient.

The current recommended chain is:

`gamma/omega -> kernel behavior -> rho / envelope / pick timing -> macro bridge -> vp/vs -> fit quality`

This means a candidate discovery should answer:

1. where in the chain the signal first appears
2. whether it survives downstream
3. whether uncertainty grows or shrinks along the chain
4. whether the relation is still present after conditioning on geometry or
   quality variables

## Next mining upgrades

The next pass should add:

1. partial correlation
2. distance/depth bucket statistics
3. bootstrap stability ranking
4. FDR-controlled screening over a wider candidate graph
5. mediation-style summaries for `rho -> pick lag -> refined_tt`
6. ablation-derived sensitivity tables for branch `gamma/omega`

## Second-pass update

The second pass is now implemented and exported in `outputs/knowledge_mining_v2`.
Compared with the first pass, it adds:

- partial Spearman correlation
- distance / depth quantile bucket summaries
- simple mediation-style chain screening
- `knowledge_bucket_panels.png`

### What changed

Most pairwise conclusions remain conservative: there is still no strong,
FDR-stable event-level law in the current 72-event screen.

However, one candidate became more interesting after conditioning on geometry
and pick error:

- `rho_p_lag -> refined_tt | distance_km, source_depth_km, pick_err_p`
  gives partial Spearman about `0.203`
- bootstrap 95% CI is approximately `[0.001, 0.420]`
- approximate `p ≈ 0.092`
- after FDR it is **not yet significant**, so it remains only a candidate

This is a useful shift in interpretation:

- plain pick lag alone was weak
- but the **P-side latent lag** (`rho_p_lag`) may contain additional
  information about inversion fit quality beyond geometry and direct pick error

That makes `rho_p_lag` a more promising causal-chain variable than raw
`rho_mean`.

### Bucket-level reading

The new bucket views also suggest two practical cautions:

1. `rho_mean` stays relatively flat across distance / depth buckets in the
   present sample
2. `vp/vs` changes across distance buckets much more visibly than `rho_mean`

So at least in this first STEAD slice:

- `rho_mean` does **not** look like a direct geometric proxy
- geometric variation may enter the final physical outputs through more complex
  timing / bridge mechanisms rather than a simple monotonic `rho_mean` trend

### Outlier warning

The nearest-distance bucket shows unusually large `refined_tt` spread and very
large mean `pick_err_p`, which likely indicates a few unstable or fallback-like
cases. Therefore:

- bucket summaries are useful descriptively
- but they should not yet be treated as stable physical laws

The next refinement should add robust trimming / winsorization or explicit
outlier tagging before stronger claims are made.

## Third-pass update

The third pass is now implemented in `outputs/knowledge_mining_v3`. It adds:

- a robust subset screen based on joint trimming of `refined_tt`,
  `pick_err_p`, and `pick_err_s`
- a lightweight multi-head comparison on the same sample pool

### Robust-subset result

Using `trim_quantile = 0.95`, the pipeline keeps `63 / 72` events and removes
the strongest tail cases.

This changes the interpretation in an important way:

- `rho_p_lag -> refined_tt | distance_km, source_depth_km, pick_err_p`
  increases from about `0.203` to about `0.248`
- approximate `p` improves from about `0.092` to about `0.054`
- bootstrap CI becomes `[-0.015, 0.483]`

This is still **not** FDR-significant, but it shows the candidate signal does
not disappear after trimming unstable rows. That makes it a more credible
candidate for follow-up.

Another new near-signal appears after trimming:

- `rho_mean -> vp_mean | distance_km, source_depth_km`
  rises to partial Spearman about `0.227`
- bootstrap CI is approximately `[0.010, 0.447]`
- approximate `p ≈ 0.079`

Again, this is not yet strong enough to claim a new law, but it suggests that
`rho_mean` may carry weak physical relevance once extreme tail cases are
removed.

### Interpretation shift after robust trimming

The current best reading is:

1. `rho_mean` is still not a direct geometric proxy
2. `rho_p_lag` remains the strongest candidate causal-chain feature for fit
   quality
3. after removing strong outliers, `rho_mean` may have a weak relation to
   recovered `vp_mean`

So the likely order of importance is now:

`rho_p_lag` first, `rho_mean -> vp_mean` second, plain `rho_mean -> geometry`
still weak.

### Multi-head comparison

The same sample pool was also used to compare several physics heads at the
init-model level:

- `zhizi_inversion_bridge_macro`
- `zhizi_inversion_bridge_residual`
- `zhizi_inversion_mixed_geo`
- `zhizi_inversion_stead_macro`

The comparison is not yet a final benchmark, but it already shows that the
heads induce noticeably different physical-output styles:

- `mixed_geo` gives the lowest average init TT among the tested heads
- `stead_macro` produces the largest init TT spread in this comparison slice
- `residual` tends to lower `vp/vs`
- `bridge_macro` and `mixed_geo` are closer in mean `vp/vs`, but not in TT

This is exactly why `gamma/omega` or macro-head knowledge should eventually be
mined **across models / checkpoints**, not only across events.
