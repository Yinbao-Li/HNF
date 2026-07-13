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

`gamma/omega -> kernel behavior -> rho / envelope / pick timing -> Physics Decoder -> vp/vs -> fit quality`

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

## Fourth-pass update (cross-model)

Pass 4 is implemented in `run_knowledge_mining_cross.py` and exported to
`outputs/knowledge_mining_v4`. It adds:

- ablation-derived sensitivity ranking from interpret-suite scans
- live stronger ablation re-check on the same event
- per-event multi-head Vp/Vs matrix + agreement
- `rho_p_lag -> refined_tt` stability across heads
- branch-0 kernel comparison across picking checkpoints (`run19/20/21`)

Figures:

- `docs/figures/knowledge/cross_head_vpvs_heatmap.png`
- `docs/figures/knowledge/live_ablation_sensitivity.png`
- `docs/figures/knowledge/ablation_sensitivity_ranking.png`

### Main conclusions

1. **Checkpoint kernels are nearly frozen across run19/20/21.**
   Branch-0 `gamma/omega/wave_speed` for P and S differ only at the 1e-4 level.
   Therefore S-only refine (`run21`) did **not** retune the branch-0 kernel
   knobs in a statistically meaningful way. Cross-checkpoint mining of
   `gamma/omega` cannot rely on these three picking checkpoints alone.

2. **Head agreement is structured, not random.**
   On the same 24-event slice:
   - `bridge_macro` ↔ `mixed_geo`: Spearman ≈ `0.82`, MAE ≈ `0.17`
   - `bridge_macro` ↔ `residual`: Spearman ≈ `0.68`, MAE ≈ `0.46`
   - `stead_macro` is the outlier vs all others (MAE ≳ `1.0`)

   This supports treating `mixed_geo` as a geo-conditioned refinement of the
   macro family, while `stead_macro` occupies a different output regime.

3. **`rho_p_lag -> refined_tt` is not head-robust.**
   Partial correlations after controlling geometry and `pick_err_p`:
   - `bridge_macro`: ≈ `+0.19`
   - `residual`: ≈ `+0.14`
   - `mixed_geo`: ≈ `-0.02`
   - `stead_macro`: ≈ `-0.27`

   Sign flips across heads. Therefore the Pass-2/3 candidate should be
   downgraded from “promising general causal-chain signal” to
   **head-specific / non-transferable candidate**.

4. **Local `gamma/omega` ablation remains weak for physical outputs.**
   Saved interpret-suite branch scans on the representative event were flat,
   partly because earlier scans wrote *effective* gamma into the *raw*
   Parameter (softplus mismatch) and only touched layer 0. That bug is now
   fixed in `run_interpret_suite.py` (raw-parameter scans over all branch
   layers). A live stronger scan still recovers only a weak
   `p_omega -> p_lag` effect (lag span ≈ `0.075 s`) and near-zero
   gamma → `vp/vs` propagation. This reinforces the architectural reading:
   branch kernel knobs mainly reshape local timing/kernel rows, not the
   macro inversion head’s velocity outputs.

### Updated knowledge status

| Claim | Status | Evidence |
|------|--------|----------|
| `rho_mean` is a geometric proxy | rejected | near-zero distance/depth correlations |
| `rho_p_lag` predicts fit quality generally | rejected as general law | fails cross-head stability |
| `mixed_geo` stays close to macro Vp/Vs | supported (descriptive) | high Spearman / low MAE |
| `stead_macro` is an outlier head | supported (descriptive) | large Vp/Vs MAE vs others |
| run21 retuned branch-0 gamma/omega | rejected | near-identical kernel params |
| local gamma/omega → vp/vs | weak / unsupported | live ablation slopes near zero |

### Next mining directions

1. Mine **head-family differences** (macro vs residual vs geo vs STEAD-only)
   with paired event deltas and bootstrap CIs, not only absolute means.
2. For `gamma/omega`, use **controlled synthetic ablations** or intentionally
   diversified checkpoints; current run19/20/21 are too close.
3. Keep event-level STEAD mining focused on timing latents and QC variables,
   but require **cross-head transfer** before claiming a new regularity.

## Paper-scale update (clustering / noise / attributes)

See `docs/PAPER_ROADMAP.md` for the full checklist. The important mining shift
after larger-N runs is:

1. **Scene clustering matters.** After robust trim (n=380), several relations
   become CI-supported globally, and some strengthen only in specific clusters.
2. **Noise-branch features help.** `noise_ratio` predicts P pick error and, in
   some clusters, init TT misfit. Mining should keep noise-cancel enabled.
3. **`rho(t)` is classically grounded.** On n=300 it correlates strongly with
   envelope / STA/LTA around P, while remaining a Huygens-conditioned latent.

## Cluster-conditioned full rediscovery

Earlier paper clustering only re-tested **4** hand-picked relations. That is
not a full rediscovery. `run_paper_cluster_rediscovery.py` re-screens a
**35-edge** candidate graph on the same robust sample:

- trim: `init_tt <= q95` → n=380
- recluster after trim with `seed=11` (matches `cluster_report_robust.*`)
- skip clusters with n<30 (C2 n=4)
- support = bootstrap CI excludes 0 **and** FDR q≤0.10

Outputs: `outputs/paper_cluster_rediscovery/` and
`docs/figures/cluster_rediscovery_summary.png`.

### Why re-run on clusters (not optional)

Whole-sample mining mixes heterogeneous STEAD scenes. A law can be:

- **global** (survives all-sample FDR+CI),
- **scene-specific** (fails globally under controls, but supported inside
  one or more eligible clusters),
- or **rejected**.

Without the cluster pass, scene-specific laws are invisible or washed out.
The previous 4-relation cluster screen was only a pilot.

### Label counts (current run)

| Label | Count |
|------|------:|
| global | 26 |
| scene-specific | 6 |
| rejected | 3 |

### Priority claims (for paper narrative)

Keep these as the main mining claims (latent / QC / causal-chain):

| Relation | Label | Notes |
|----------|-------|-------|
| `rho_p_lag -> init_tt` (partial) | global | ≈ −0.41; also in C1 |
| `rho_mean -> vp_mean` (partial) | global | ≈ −0.29; stronger in C0/C3 |
| `noise_ratio -> pick_err_p` (partial) | global | ≈ +0.17 |
| `noise_ratio -> init_tt` (partial) | scene-specific | C1/C3 positive; pairwise global sign flips |
| `rho_p_lag -> vp_mean` (partial) | scene-specific | C3 only |

Downgrade / interpret cautiously:

- very strong `rho_mean -> vpvs_mean` (≈ 0.73): likely **head-induced coupling**
  until cross-head transfer is shown
- geometry → `vp_mean` / `rho_mean` edges: expected geo conditioning, not new
  Huygens physics

### Cross-head transfer (priority table)

Script: `run_paper_cross_head_transfer.py` → `outputs/paper_cross_head_transfer/`
(n=200 head-forward events × 4 heads; QC law on full n=380).

| Relation | Transfer label |
|----------|----------------|
| `rho_p_lag -> init_tt` | **head_robust** (all 4 heads, negative) |
| `rho_mean -> vp_mean` | **sign_unstable** (macro + vs mixed/stead −) |
| `rho_mean -> vpvs_mean` | head_robust* (3/4; residual Vp/Vs constant) |
| `noise_ratio -> pick_err_p` | **head_independent_supported** |
| `noise_ratio -> init_tt` | head_specific_or_weak on all-sample slice |

Paper-facing keepers: `rho_p_lag→init_tt`, `noise_ratio→pick_err_p`.
Downgrade `rho_mean→vp/vpvs` to head-family descriptive couplings.

### Absolute geography (lat/lon) rediscovery + confirmation

Scripts:

- `run_paper_geo_rediscovery.py` → `outputs/paper_geo_rediscovery/`
- `run_paper_geo_confirm.py` → `outputs/paper_geo_confirm/`

Figures: `docs/figures/geo_cluster_map.png`, `geo_qc_spatial_map.png`,
`geo_priority_controls.png`, `geo_absolute_vs_network.png`,
`geo_sensitivity_heatmap.png`.

Sample context (critical for interpretation):

- n=380 after the same robust `init_tt` trim
- networks: **ZQ 310**, TA 66, others 4 — not a California-only slice
- geo-kmeans is unbalanced (C3≈309); lon tertiles are balanced (~127)

| Claim | Confirmation label | Evidence |
|-------|--------------------|----------|
| `noise_ratio → pick_err_p` | **CONFIRMED (strong)** | partial ≈ +0.16 with lat/lon controls; ≈ +0.16 with `is_ZQ`; holds in ZQ-only |
| `rho_p_lag → init_tt` | **CONFIRMED (strong)** | partial ≈ −0.38 with lat/lon; ≈ −0.39 with `is_ZQ`; holds in ZQ-only |
| `rho_mean → vp_mean` | CONFIRMED (moderate) | survives lat/lon and `is_ZQ` here; still head-unstable from cross-head pass |
| `source_lat → pick_err_p` | **REINTERPRETED** | pairwise/partial vs dist/depth supported; **collapses after `is_ZQ`** |
| `source_lon → pick_err_p` | geo / within-ZQ | global often geo-specific (C3); **within ZQ** partial ≈ +0.15 still supported |
| `is_ZQ → pick_err_p` / `noise_ratio` | supported confounder | network tag explains much of the absolute-geo QC signal |

**Physical / operational explanation (not overclaim):**

1. Absolute lat/lon are informative because the mining table mixes distinct
   acquisition regions (ZQ lobe vs TA/other). That induces differences in
   noise field, path complexity, and instrument/network practice that show up
   in `pick_err_*`, `noise_ratio`, and TT fit — even after controlling
   epicentral distance and depth.
2. Therefore lat/lon edges should be narrated as **regional/network geography**,
   not as a universal latitude–error law.
3. Priority Huygens latents (`noise_ratio`, `rho_p_lag`) are **not** explained
   away by that geography: they remain after lat/lon and after `is_ZQ`, and
   remain inside ZQ-only. That is the paper-facing geo confirmation.
4. Sensitivity on leave-C3 / non-ZQ (n≈70) often loses FDR support; treat as
   **low power**, not as a rejection of the global laws.

### Updated next steps

1. Keep scene labels **and** network / lat–lon in any future STEAD mining export
2. Do not claim scene-specific laws from C2-sized clusters
3. Always report absolute-geo claims with a network (`is_ZQ`) or region control
4. External-dataset Fig5 still blocked (no Instance/DiTing loader/data)
