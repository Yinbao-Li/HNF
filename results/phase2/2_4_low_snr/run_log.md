# Phase B / 2.4 run log

1. Cond1 diagnosis (`diagnosis_proposal_vs_refine.json`): −5 dB P proposal-dominated → proceed.
2. Pre-registration locked in `PREREGISTER.md` **before** train/eval.
3. Train: `scripts/phase2_2_4_proposal_head.sh` → `outputs/stead_c2f_2_4_proposal_head/` (~17 min, 6 ep, proposal-head-only + SNR-aware).
4. Eval: `snr_curve_test.json`, `snr_curve_val_clean_check.json`, `prop1_verify.json`.
5. Primary Δ P F1 @ −5 dB = **+0.016** → 未修复 (&lt; +0.05).
6. Prop1: ethz_dev P **created=1** → rollback SNR-aware code from `picking_coarse_to_fine.py`; ckpt not shipped.
7. Artifacts: `RESULTS.json`, `CONCLUSION.md`. Phase C not started.
