# Phase 2 live status (2026-11-16)

## Completed (pushed)
- Spec, 2.7 (AE+headline CI), 2.5, 2.3B PickBlue, 2.3A→Path C, 2.4 design, 2.6 reject

## Running
- **2.2** matched-budget EQT→PN to 182 ep (`results/phase2/2_2_matched_budget/`)
  - EQT ~230 s/ep → ~11.6 h; then PN ~8 h; then auto-eval
  - ETA training ~20 h from 13:01 local

## Queued (auto after 2.2)
- **2.1** pair × seeds 42(reuse)+43+44 (`scripts/phase2_2_1_pair_seeds.sh`)
  - ~9.2 h/run × 4 new runs ≈ 37 h

## Quote only (not run)
- 80-ep trunk fork multi-seed: ~178 GPU-h; measures late-stage variance only
  (`results/phase2/2_1_pair_seeds/QUOTE_80ep_fork.md`)

## Blocked / closed
- SeisT download/eval: Path C (DiTing weights ≠ STEAD-only)

## Not entering Phase 3 until 2.2+2.1 finish and Phase 2 summary is written.
