# Phase 2 summary — HNF TGRS v14 (stop; do not enter Phase 3)

Date closed: 2026-11-17  
Spec: `docs/TGRS_TARGET_SPEC.md` (verbatim). **No `main.tex` edits in Phase 2.**

## Item status

| Item | Status | Artifacts |
|------|--------|-----------|
| Spec | DONE | `docs/TGRS_TARGET_SPEC.md` |
| **2.7** Cluster CI | DONE | `results/phase2/2_7_ci/` |
| **2.5** Controlled compute | DONE | `results/phase2/2_5_compute/` |
| **2.3B** PickBlue 4C ID | DONE | `results/phase2/2_3b_pickblue_obs/` |
| **2.3A** SeisT | FAIL → **Path C** | `results/phase2/2_3a_seist_verify/` (no download/eval) |
| **2.2** Matched budget | **REVISED** | Align at **20 ep** via existing `outputs/sb_aug_{eqt,phasenet}`; 182-ep aborted (`2_2_matched_budget/STOPPED.md`) |
| **2.1** Pair × 3 seeds | DONE | `results/phase2/2_1_pair_seeds/` |
| **2.4** Low-SNR fix | Design only | `results/phase2/2_4_low_snr_design.md` — **no experiment until approved** |
| **2.6** ρ-scrub scale | REJECTED | `results/phase2/2_6_rejected.md` |

## Key numbers vs manuscript (report only)

### 2.5 Table V (controlled re-bench)
Throughput **−8.7% … −12.5%** vs frozen manuscript; MiB essentially match. Continuous bf16 win/s recorded. Do not overwrite Table V without protocol disclosure.

### 2.7 AE cluster CI (ours − pn_aug, event)
| Material | Tol | ΔF1 | Event CI excl0? | ΔAP event excl0? |
|----------|-----|-----|-----------------|------------------|
| Minusil | 10/20 | + | YES | YES |
| Minusil | **50** | **−0.023** | YES (against ours) | YES (AP still +) |
| Glass | 10/20/50 | + | YES | YES |

Headline STEAD dump P@0.5 CI **excludes** manuscript 0.988; ETHZ/OBS dump abs F1 ≪ paper `eval_ethz_cross` — treat dump absolute F1 as **suspect** until dump path matches paper decoder.

### 2.3B PickBlue 4C (separate ID table)
PN-base @0.5: P/S **0.924 / 0.849** (n=6932 4C). Not comparable to paper 3C STEAD→OBS rows (n=11256).

### 2.3A → Path C
Published SeisT dpk weights are **DiTing**-trained → cannot fill “STEAD-only modern land picker” row.

### 2.1 Pair 20 ep × seeds 42/43/44 (mean±std)
| Arm | STEAD val P/S | ETHZ P/S @0.5 | OBS P/S @0.5 |
|-----|---------------|---------------|--------------|
| huygens | 0.979±0.003 / 0.962±0.001 | 0.959±0.001 / 0.972±0.001 | 0.628±0.005 / 0.732±0.003 |
| free | 0.982±0.002 / 0.974±0.002 | 0.957±0.002 / 0.974±0.000 | 0.631±0.006 / 0.751±0.004 |

Δ(huygens−free) mean: STEAD P/S **−0.003 / −0.012**; ETHZ **+0.003 / −0.003**; OBS **−0.003 / −0.020**.

### 2.2 (revised)
No new training. Manuscript fairness note should state EQT-aug/PN-aug are **matched at 20 ep** (same aug recipe), not at main-model ~182 ep. Do **not** claim longer-budget parity.

## Phase 3 claim actions (wording only; not started)

1. **Drop / rewrite** Abstract universal “highest sub-0.1 s share … Ridgecrest/OBS” (already AUDIT UNSUPPORTED).
2. **Scope AE F1 lead**: not at Minusil tol50; prefer AP or tight-tol framing that survives event CI.
3. **Kernel / Huygens vs free**: at 20-ep multi-seed, free ≈ or slightly better on F1 — do **not** claim Huygens accuracy superiority; keep “equals on quake metrics / prior for auditability” only if scoped to Prop 1 + interventions, not F1 win.
4. **No SeisT STEAD-only baseline row** (Path C); PickBlue only as labelled 4C OBS ID reference.
5. **Table V**: disclose controlled re-bench Δ or keep frozen numbers with hardware note.
6. **36 scalars → 45**; Mars/template-matching → supplement colour.
7. **2.4**: only after design approval + Prop 1 re-verify (0 created picks).
8. **2.6**: not claims.

## Explicit non-actions
- Did not edit `paper/tgrs/main.tex`.
- Did not start Phase 3 rewrite.
- Did not run 2.4 experiments.
- Did not scale 2.6.

## Superseded by Phase A (2026-11-17) — do not delete history above

- ~~Headline ETHZ/OBS dump abs F1 ≪ paper / dump path suspect~~ → **FIXED**: dump now matches `run_c2f`+`metrics_at`; see `2_7_ci/CORRECTION.md`.
- ~~STEAD P@0.5 CI excludes manuscript 0.988~~ → **FIXED**: CI [0.986, 0.990] includes 0.988.
- AE Minusil tol50 ΔF1 = −0.023 excl0 → **CONFIRMED unchanged**.
- Manuscript STEAD 0.988 → **confirmed correct** (not a manuscript bug).
