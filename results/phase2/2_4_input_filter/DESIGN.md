# Input filter → frozen c2f picker (design)

## Goal

Train a **waveform front-end filter** \(F_\phi\), then run the **frozen** paper picker \(M\) (`imp_RSP`) on the filtered window:

\[
\hat{y} = M\big(F_\phi(x)\big)
\]

Not a residual-of-refine interpreter. Aim: improve low-SNR proposal recall without breaking Prop 1 / clean-SNR.

## Architecture

- **Input:** same as c2f — \((B,T,C)=(B,6000,3)\), demean/std **before** filter (manuscript normalize).
- **Filter \(F_\phi\):** residual Conv1d stack, near-identity init:
  - \(x' = x + \alpha\,\mathrm{CNN}(x)\), \(\alpha\) softplus-gated, init CNN≈0 so \(F\approx\mathrm{id}\) at start.
  - Depthwise/grouped or small hidden (e.g. 16–32 ch), k=7/15, 3–5 layers; **no time-warp / resampling**.
  - Optional: zero-mean per channel constraint on CNN output (do not inject DC).
- **Picker \(M\):** load `outputs/stead_c2f_imp_RSP/last.pt`, **all weights frozen**.
- **Prop 1:** unchanged (refine still gated by physics proposal inside \(M\)).

## Training (pilot → full)

1. Freeze \(M\); train only \(\phi\).
2. Loss = same pick/det/wave-sup as c2f train (reuse `train_stead_c2f_picking` loss path) on \(M(F(x))\).
3. Aug: emphasize real-noise incl. −5 dB (same family as 2.4), keep clean mix so identity path stays good.
4. Select on STEAD val clean pick_focus + val −5 dB P F1 (if cheap) — **no test tuning**.

## Success / stop (pre-register for any full run)

Same spirit as Phase B:
- Primary: test SNR −5 dB P F1 vs `imp_RSP` baseline (~0.453); **fix** if Δ≥+0.10 and clean/high-SNR regress ≤0.01.
- **未修复** if Δ&lt;+0.05 or head damage or Prop1 created&gt;0.
- Budget: pilot ≤1 GPU·h; full ≤~6 GPU·h then stop/report.

## Non-goals

- Do not unfreeze backbone in pilot.
- Do not edit `main.tex` from pilot alone.
- Do not claim “new physical law”; claim is engineering front-end if metrics move.
