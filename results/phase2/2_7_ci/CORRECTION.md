# Phase A — 2.7 dump vs manuscript decoder (2026-11-17)

## Verdict

**Dump path was wrong; manuscript STEAD headline numbers are OK.**  
Absolute F1 collapse on ETHZ/OBS (and STEAD CI excluding 0.988) was **not** a manuscript error.

## Root causes (evidence)

Same checkpoint `outputs/stead_c2f_imp_RSP/last.pt`, same windows (seed=0 subset), ablation in
`diagnose_{ethz,obs}_n800.json`.

| Factor | Manuscript (`eval_ethz_cross` → `run_c2f` + `metrics_at`) | Broken 2.7 dump | Effect |
|--------|-----------------------------------------------------------|-----------------|--------|
| Input normalize | demean + per-channel **std** | **none** (raw after bandpass) | **Primary** |
| Time axis | `linspace(0,60,T)` shape `(B,T,1)` | `arange*(60/T)` shape `(B,T)` | Negligible once normalized |
| Pick extract | argmax on gated probs; `metrics_at` applies `p_before_s` | argmax after dump-side `p_before_s` | Minor |
| F1 counting | EQT-style: event wrong-pick = **FN only** | dump: wrong-pick = **FP + FN** | **Secondary** (esp. S) |
| Channels | OBS/ETHZ 3C as in `load_dataset` (OBS `21Z`) | same cut_window / bandpass | Match |
| Labels | earliest P/S cols; P placed in [5,30] s | same | Match |

### Same-subset numbers (ETHZ n=745)

| Decoder variant | metrics_at P/S @0.5 | dump-style F1 P/S @0.5 |
|-----------------|---------------------|-------------------------|
| `ms_run_c2f` (manuscript) | **0.956 / 0.985** | 0.948 / 0.720 |
| `dump_raw_no_norm` (old 2.7) | **0.679 / 0.332** | 0.671 / 0.215 |
| `dump_plus_norm_flat_t` | 0.956 / 0.985 | 0.948 / 0.721 |

Per-sample (tol0.5 P): manuscript hit rate 0.915 vs dump-raw 0.514; **ms_hit∧dump_miss = 0.405**.

OBS n=655: manuscript metrics_at P@0.5 **0.618** vs dump-raw **0.227** — same pattern.

### STEAD

Dataset already demean/std-normalizes → primary bug absent. Remaining gap was dump-style F1 + incomplete comparison. After metrics_at counts: full-test **P@0.5 = 0.98815** (manuscript 0.988).

## Fix

`tools/phase2_dump_headline_hits.py` now:
1. ETHZ/OBS: `run_c2f` + per-trace counts from `update_picking_counts` semantics  
2. STEAD: dataset batches + same count rules  
3. Bootstrap reads `counts` arrays (legacy hit/pred/valid still parsed)

Old dumps archived under `pre_correction/`.

## Before → after (imp_RSP)

| Set | Metric | Pre-correction dump | Corrected | Manuscript / verify |
|-----|--------|---------------------|-----------|---------------------|
| STEAD | P@0.5 | 0.9825 (CI excl 0.988) | **0.9881** CI [0.986, 0.990] **incl 0.988** | 0.988 |
| STEAD | S@0.5 | 0.9581 | **0.9751** | 0.975 |
| ETHZ | P@0.5 | 0.6639 | **0.9505** | `eval_ethz_cross` **0.9505** (match) |
| ETHZ | S@0.5 | 0.2073 | **0.9768** | `eval_ethz_cross` **0.9768** |
| OBS | P@0.5 | 0.2576 | **0.6321** | (imp_RSP; paper `final` row 0.556 was other ckpt) |
| OBS | S@0.5 | 0.1474 | **0.7752** | |

Note: `outputs/.../final/ethz_cross.json` “final” P@0.5≈0.897 is a **different checkpoint label**, not imp_RSP. Corrected dump matches fresh `eval_ethz_cross` on **imp_RSP**.

## AE re-check (unchanged by dump fix)

Minusil tol50 ΔF1 ours−pn_aug = **−0.023**, event CI excl 0 (**still stands**).  
AP event CI still favours ours at tol50. Glass all tols still +excl0.

## Phase 2 conclusions changed

| Prior Phase 2 note | After correction |
|--------------------|------------------|
| “ETHZ/OBS dump abs F1 suspect / do not use” | **Superseded** — corrected dumps usable; match manuscript decoder |
| “STEAD CI excludes manuscript 0.988” | **Superseded** — CI includes 0.988 |
| Minusil tol50 F1 reverse | **Unchanged** — still true |
| Manuscript STEAD 0.988 wrong? | **No** |
