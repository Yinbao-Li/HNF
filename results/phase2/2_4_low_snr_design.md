# Phase 2.4 — Low-SNR proposal failure: design only (no runs)

Status: **DESIGN SUBMITTED — await approval before any experiment.**

## Failure to fix

At low SNR (manuscript / AUDIT: e.g. −5 dB P F1 ≈ 0.45 vs PN-aug ≈ 0.76),
misses are dominated by **missing or wrong coarse proposals**, not by refine
mis-centering inside a correct window. Created picks must remain impossible
under Proposition 1 (bounded CNN refine cannot invent picks outside proposals).

## Mechanism to change (proposal formation only)

Target: the **coarse proposal generator** upstream of refine — specifically how
candidate onsets are scored/thresholded under high noise — **not** the refine
CNN capacity, loss, or R-bound.

Proposed change (single lever, keep Prop 1 intact):

1. **SNR-aware proposal score:** replace or gate the current coarse peak score
   with a noise-normalized statistic (e.g. local SNR or MAD-normalized STA/LTA
   on the same coarse features already used for proposals), so weak but
   coherent onsets remain above the propose threshold when absolute amplitude
   collapses.
2. **Optional:** slightly lower propose threshold **only when** the noise-normalized
   score exceeds a floor — still emit a discrete proposal set; refine stays
   inside ±R of those proposals (no free decoding on the full trace).

Out of scope for this fix: enlarging R, adding a parallel free decoder,
waveform editing, or ρ-scrub.

## Why this should help

Low-SNR FNs are proposal absences. Improving proposal recall at low SNR
directly raises P/S hit rate without giving refine authority to create picks.
If the diagnosis is wrong (refine already receives good proposals), the SNR
curve will not move — that is a clean falsification.

## Proposition 1 preservation plan (mandatory)

After any code change and before claiming a fix:

1. Re-run the existing **7,000-window** Prop 1 verification suite
   (same protocol as manuscript: refine cannot create picks outside proposals).
2. **Acceptance:** still **0 created picks**.
3. If any created pick appears → **revert**; do not ship the change.
4. Log verification JSON under `results/phase2/2_4_low_snr/` with
   `n_created_picks`, window ids, and code git SHA.

## Eval plan (only after design approval)

1. Implement proposal-only patch on a branch from `tgrs-v14` / `imp_RSP`.
2. Short FT from `stead_c2f_imp_RSP/last.pt` (est. 6–12 ep) **or** freeze backbone
   and only train proposal head — prefer the smaller surface.
3. Re-run SNR curve vs PN-aug / EQT-aug under the same protocol as Fig. snr.
4. Prop 1 re-verify (7000 windows, 0 created).
5. Report deltas vs manuscript numbers only; **do not edit main.tex** in Phase 2.

## Explicit non-goals

- No main.tex edits.
- No Prop 1 relaxation.
- No 2.6-style waveform interventions as a substitute.
EOF
