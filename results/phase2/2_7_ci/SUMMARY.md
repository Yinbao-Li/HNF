# Phase 2.7 cluster-bootstrap CI summary

Protocol: F1 n_boot=1000 (trace / event / sensor); AP n_boot=200 (event only). Seeds 0–4. Ref = ours vs pn_aug.

## AE (primary): event-cluster vs paired-trace

| Material | Tol | ΔF1 ours−pn_aug | Trace CI excl0? | Event CI excl0? | ΔAP event CI excl0? |
|---|---:|---:|---|---|---|
| Minusil_test | tol10 | +0.056 | True [0.051, 0.061] | True [0.050, 0.062] | True [0.090, 0.102] |
| Minusil_test | tol20 | +0.010 | True [0.005, 0.014] | True [0.005, 0.014] | True [0.053, 0.062] |
| Minusil_test | tol50 | −0.023 | True [−0.027, −0.020] | True [−0.027, −0.019] | True [0.032, 0.041] |
| Glass_beads_test | tol10 | +0.082 | True [0.066, 0.098] | True [0.065, 0.098] | True [0.064, 0.102] |
| Glass_beads_test | tol20 | +0.043 | True [0.029, 0.056] | True [0.030, 0.057] | True [0.036, 0.073] |
| Glass_beads_test | tol50 | +0.030 | True [0.020, 0.040] | True [0.020, 0.041] | True [0.047, 0.070] |

## Claim status (AE physics-prior / AE lead)

| Claim scope | Survives event-cluster CI? | Note |
|---|---|---|
| Ours > pn_aug F1 @ tol10/20 (both materials) | **YES** (excl 0) | Keep |
| Ours > pn_aug F1 @ Glass tol50 | **YES** | Keep |
| Ours > pn_aug F1 @ Minusil tol50 | **NO — reverse, excl 0** | Must **downgrade / drop** universal “AE F1 lead at all tols”; scope to tight tol or AP |
| Ours > pn_aug AP (event) all listed tols | **YES** (excl 0, even Minusil tol50) | Prefer AP-framed AE claim if F1@50 kept soft |
| Trace vs event CI disagreement | None material for headline ΔF1 | Event clustering does not overturn prior paired-trace story on these rows |

## Headline tables (STEAD / ETHZ / OBS)

- Status at AE-finish: dumps **MISSING** — `phase2_dump_headline_hits.py` not yet run; re-bootstrap after dumps land.
- Files expected: `stead_full_hits.json`, `ethz_test_hits.json`, `obs_test_hits.json`.

## Artifacts

- `ae_cluster_bootstrap.json` — full CIs (all models, all schemes)
- `headline_cluster_bootstrap.json` — placeholder until dumps exist
