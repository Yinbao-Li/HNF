# Phase 2.3 Path B — PickBlue 4C OBS in-distribution reference

**Not a cross-domain baseline.** Separate from STEAD→OBS transfer tables.

- Split: SeisBench OBS `test` (event-holdout; ID for PickBlue)
- Input: **4C Z12H** (H high-pass 0.5 Hz); paper transfer rows use **3C 21Z**
- Windows: 6932 (S-valid 3152); dead-ch dropped 0
- Protocol: P in [5.0, 30.0] s; bandpass Z12=1,45; th=0.3/det=0.5

| Model | Tol | P F1 | S F1 |
|---|---:|---:|---:|
| PickBlue(phasenet)-4C-Z12H | 0.1 | 0.7942 | 0.6393 |
| PickBlue(phasenet)-4C-Z12H | 0.2 | 0.8771 | 0.7586 |
| PickBlue(phasenet)-4C-Z12H | 0.5 | 0.9237 | 0.8488 |
| PickBlue(eqtransformer)-4C-Z12H | 0.1 | 0.7763 | 0.5481 |
| PickBlue(eqtransformer)-4C-Z12H | 0.2 | 0.8773 | 0.6664 |
| PickBlue(eqtransformer)-4C-Z12H | 0.5 | 0.9306 | 0.7324 |

## Disclosures (must travel with any table)
1. In-distribution for PickBlue (OBS-trained, same corpus holdout).
2. 4C vs paper 3C — not matched input dimensionality.
3. Do not fold into STEAD-only transfer comparison.

## vs paper 3C STEAD→OBS transfer (do not merge)

Paper `final/obs/obs_test.json` uses **3C 21Z**, n_windows=**11256**.
This Path B table uses **4C Z12H** complete traces only, n=**6932** (subset requiring H).
Numbers are **not** matched-set comparable; ID ceiling only.

Paper 3C transfer (illustrative, same tol protocol, different channel set/n):
| Model (3C transfer) | Tol0.1 P/S | Tol0.5 P/S |
|---|---|---|
| final (ours) | 0.452 / 0.513 | 0.556 / 0.781 |
| EQT(STEAD) | 0.424 / 0.448 | 0.612 / 0.721 |
| PhaseNet(STEAD) | 0.291 / 0.458 | 0.435 / 0.707 |

PickBlue 4C ID (this run): PN-base 0.794/0.639 @0.1; 0.924/0.849 @0.5 — expected higher as OBS-trained ID ceiling.
