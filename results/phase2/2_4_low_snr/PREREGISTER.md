# Phase B / 2.4 — pre-registration (before any fix train or eval)

Written: 2026-11-17 after Cond1 diagnosis; **before** code change / training / SNR eval.

## Cond1 gate (frozen `imp_RSP`, STEAD val n=3000, seed=7, tol=0.5 s)

Source: `diagnosis_proposal_vs_refine.json`

| SNR | Phase | proposal_recall_local | miss_no_proposal | miss_with_proposal | refine_ecc median (s) | verdict |
|-----|-------|----------------------:|-----------------:|-------------------:|----------------------:|---------|
| −5  | P     | 0.318 | 0.681 | 0.036 | 0.040 | **proposal_dominated** |
| −5  | S     | 0.585 | 0.414 | 0.060 | 0.060 | proposal_dominated |
| 0   | P     | 0.654 | 0.345 | 0.063 | 0.050 | proposal_dominated |

**Proceed:** yes — (a) dominates (b) at −5 dB P. Refine arm not touched.

## Noise-normalized proposal score (exact definition)

On physics proposal logits `w ∈ R^{B×T}` **after** `wave_bias`, **before** refine max-pool / gate:

1. `med = median_t(w)`
2. `mad = median_t(|w − med|)`
3. `z_t = (w_t − med) / (1.4826 · mad + MAD_FLOOR)`

**Threshold relief (design optional #2), soft form:**

`w ← w + BOOST · σ( K · (z − Z_FLOOR) )`

Equivalent intent: when `z` is a clear MAD outlier, behave as if pick threshold were lowered from 0.3 → 0.15 on that bin (BOOST = logit(0.3)−logit(0.15)).

### Frozen constants (not to be tuned on test)

| Symbol | Value | Role |
|--------|------:|------|
| `MAD_FLOOR` | 0.05 | logit MAD floor |
| `Z_FLOOR` | 2.0 | require ~2σ_MAD outlier |
| `BOOST` | 0.84729786 | logit(0.3)−logit(0.15) |
| `K` | 4.0 | soft-step sharpness |
| `gate_floor` ε | 0.0 | **unchanged** (Prop 1) |
| `refine_max` R | 4.0 | **unchanged** |
| `refine_pool_sec` | 0.2 | **unchanged** |

## Quantities that change

- **Changed:** `p_wave` / `s_wave` (SNR-aware boost) → hence fused `p`/`s` via existing propose_refine formula.
- **Trained (proposal-head-only):** `p_wave_head.*`, `s_wave_head.*`, `wave_bias`.
- **Frozen:** encoder, Huygens layers, fuse_*, time/skip/det heads, refine path, R, ε, pool.

## Training protocol (pre-registered)

- Init: `outputs/stead_c2f_imp_RSP/last.pt`
- Epochs: 6; `samples_per_epoch=100000`; lr=1e-4; cosine; warmup=0
- Aug: same family as imp_RSP, but `aug_real_noise_prob=0.8`, `aug_real_noise_snr_min=-5`, `aug_real_noise_snr_max=10`
- Loss emphasis: `wave_sup_weight=1.0`, `wave_pos_weight=3.0`; refine_sup kept for stability but refine params frozen (no grad)
- Select: STEAD **val** shifted pick_focus (no test peek)

## Eval protocol (pre-registered; no post-hoc metric edits)

1. SNR curve: same as `outputs/paper_tgrs/launch_snr_test.sh` (STEAD **test**, seed=7, n_ev=3000, n_nz=1000, th 0.3/0.5, tol 0.1 & 0.5) vs `eqt_aug` / `pn_aug`.
2. Per SNR: report **P/S precision and recall** (tol 0.5 primary).
3. Clean-SNR regression: STEAD **val** clean (and full SNR on val optional) — head F1 drop ≤ 0.01 vs frozen imp_RSP.
4. Prop 1: `analyze_refine_role.py` stead_val (3k+1k) + ethz_dev (3k) = 7k windows; require **0 created picks**.

## Success criteria (locked)

- Primary: −5 dB **P F1** (tol 0.5, test SNR curve) vs manuscript ours ≈ 0.453.
- **修复成功:** Δ ≥ +0.10 **and** high-SNR / STEAD·ETHZ head F1 regress ≤ 0.01 **and** not precision-bought (no large FP trade) **and** Prop1 = 0 created.
- **未修复:** Δ < +0.05 **or** head damage **or** FP-bought **or** Prop1 fail (revert).
- Mid (+0.05…+0.10): list numbers; wait for user adjudication.
- Budget: ≤ ~12 GPU·h (3080 Ti); stop if exceeded.
- No `main.tex` edits. Stop after Phase B report — do not start Phase C.
