# Phase 2.7 cluster-bootstrap CI summary

Protocol: F1 n_boot=1000 (trace / event / sensor); AP n_boot=200 (event only). Seeds 0–4. Ref = ours vs pn_aug.
Headline dumps: `*_hits.json` (ours only); station/station-prefix clustering.

## AE (primary): event-cluster vs paired-trace

| Material | Tol | ΔF1 ours−pn_aug | Trace CI excl0? | Event CI excl0? | ΔAP event CI excl0? |
|---|---:|---:|---|---|---|
| Minusil_test | tol10 | +0.056 | True [0.051, 0.061] | True [0.050, 0.062] | True [0.090, 0.102] |
| Minusil_test | tol20 | +0.010 | True [0.005, 0.014] | True [0.005, 0.014] | True [0.053, 0.062] |
| Minusil_test | tol50 | −0.023 | True [−0.027, −0.020] | True [−0.027, −0.019] | True [0.032, 0.041] |
| Glass_beads_test | tol10 | +0.082 | True [0.066, 0.098] | True [0.065, 0.098] | True [0.064, 0.102] |
| Glass_beads_test | tol20 | +0.043 | True [0.029, 0.056] | True [0.030, 0.057] | True [0.036, 0.073] |
| Glass_beads_test | tol50 | +0.030 | True [0.020, 0.040] | True [0.020, 0.041] | True [0.047, 0.070] |

### AE claim status

| Claim | Survives event CI? | Phase 3 action |
|---|---|---|
| Ours > pn_aug F1 @ tol10/20 (both) | YES | Keep |
| Ours > pn_aug F1 @ Glass tol50 | YES | Keep |
| Ours > pn_aug F1 @ Minusil tol50 | **NO (reverse, excl 0)** | Downgrade / drop universal AE F1 lead |
| Ours > pn_aug AP (event) all listed | YES | Prefer AP-framed AE claim |

## Headline tables (ours point F1 + cluster CI)

| Set | Metric | Dump F1 | Cluster CI95 | Manuscript | Δ dump−MS |
|---|---|---:|---|---:|---:|
| STEAD full | P@0.1 | 0.8957 | [0.878, 0.912] | 0.942 | −0.046 |
| STEAD full | P@0.5 | 0.9825 | [0.979, 0.985] | 0.988 | −0.0055 |
| STEAD full | S@0.1 | 0.7734 | [0.739, 0.807] | 0.869 | −0.096 |
| STEAD full | S@0.5 | 0.9581 | [0.949, 0.965] | 0.975 | −0.017 |
| ETHZ test | P@0.5 | 0.6639 | [0.636, 0.691] | (see final/ethz) | dump≪paper |
| OBS test | P@0.5 | 0.2576 | [0.224, 0.291] | paper final≈0.556 | dump≪paper |

### Headline notes (do not edit main.tex)

1. STEAD P@0.5 dump CI **excludes** manuscript 0.988 → protocol/decoding mismatch or different subset; report only, no number rewrite in Phase 2.
2. ETHZ/OBS dump F1 far below `eval_ethz_cross.py` paper JSON → dump path still suspect for cross-domain; **do not** use these absolute F1 as claim evidence until dump matches `run_c2f` paper numbers. CIs above are conditional on dump hits.
3. No multi-model headline dumps yet (EQT/PN) — only ours.

## Artifacts

- `ae_cluster_bootstrap.json`, `headline_cluster_bootstrap.json`
- `stead_full_hits.json`, `ethz_test_hits.json`, `obs_test_hits.json`
