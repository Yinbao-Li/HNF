#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rheology tube–MWD alignment under per-sample vs global vs LOO anchoring.

Per-sample anchoring (manuscript primary to date) uses each sample's own GPC
median to place λ→M. That is shape alignment after target-side position
calibration, not a blind MWD readout.

This script compares:
  - per_sample: current tube_corr (uses that sample's GPC median)
  - global: fit a on all 9 via logM_med ≈ a + α^{-1} logλ_med; apply to all
  - loo: fit a on 8; evaluate tube corr on held-out sample without its GPC median
  - fixed_literature: a chosen so mean predicted logM matches literature Me scale
    (optional weak baseline: a = 4.5 - α^{-1}*mean_logλ across library)

Also reports a cohort-level mean Δ = mean(r_obs - r_null) with sample bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from hnf.rheo_freq_fit import default_lambda_grid, fit_prony_freq_pnf
from hnf.rheo_gpc import load_leeds_gpc_all, mwd_on_log_grid
from hnf.rheo_leeds import load_leeds_saos_all
from tools.run_rheo_leeds_tube_harden import BRANCHED, shuffle_null

ALPHA = 3.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--output-dir", default="outputs/rheo/loo_anchoring")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--n-shuffle", type=int, default=800)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _gpc_logM_med(logM_grid, mwd) -> float:
    mwd = np.maximum(np.asarray(mwd, dtype=np.float64), 0.0)
    return float(np.average(logM_grid, weights=np.maximum(mwd, 1e-12)))


def _w_loglam_med(lam, g) -> float:
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    w = g / max(g.sum(), 1e-30)
    lam = np.asarray(lam, dtype=np.float64)
    return float(np.average(np.log10(np.maximum(lam, 1e-30)), weights=w))


def tube_corr_anchored(lam, g, logM_grid, mwd, alpha: float, logM_anchor: float) -> float:
    """Align using a supplied logM anchor (not recomputed from this sample's GPC)."""
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    lam = np.asarray(lam, dtype=np.float64)
    w = g / max(g.sum(), 1e-30)
    mwd = np.asarray(mwd, dtype=np.float64)
    log_lam_med = float(np.average(np.log10(np.maximum(lam, 1e-30)), weights=w))
    logM_of_lam = logM_anchor + (1.0 / alpha) * (np.log10(np.maximum(lam, 1e-30)) - log_lam_med)
    # When anchor is an intercept a with logM = a + α^{-1} log10(λ):
    # equivalent form used when logM_anchor_mode == 'intercept' via caller.
    mwd_mass = np.zeros(len(lam))
    dlog = float(logM_grid[1] - logM_grid[0]) if len(logM_grid) > 1 else 1.0
    for j, lm in enumerate(logM_grid):
        k = int(np.argmin(np.abs(logM_of_lam - lm)))
        mwd_mass[k] += mwd[j] * dlog
    mwd_mass = mwd_mass / max(mwd_mass.sum(), 1e-30)
    if w.std() < 1e-12 or mwd_mass.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(w, mwd_mass)[0, 1])


def tube_corr_intercept(lam, g, logM_grid, mwd, alpha: float, a: float) -> float:
    """log10 M = a + α^{-1} log10 λ  (no per-sample GPC median)."""
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    lam = np.asarray(lam, dtype=np.float64)
    w = g / max(g.sum(), 1e-30)
    mwd = np.asarray(mwd, dtype=np.float64)
    logM_of_lam = a + (1.0 / alpha) * np.log10(np.maximum(lam, 1e-30))
    mwd_mass = np.zeros(len(lam))
    dlog = float(logM_grid[1] - logM_grid[0]) if len(logM_grid) > 1 else 1.0
    for j, lm in enumerate(logM_grid):
        k = int(np.argmin(np.abs(logM_of_lam - lm)))
        mwd_mass[k] += mwd[j] * dlog
    mwd_mass = mwd_mass / max(mwd_mass.sum(), 1e-30)
    if w.std() < 1e-12 or mwd_mass.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(w, mwd_mass)[0, 1])


def fit_intercept_a(rows: list[dict], alpha: float) -> float:
    """Fit a from logM_med ≈ a + α^{-1} logλ_med over training samples."""
    ys = []
    xs = []
    for r in rows:
        ys.append(r["logM_med"])
        xs.append(r["loglam_med"] / alpha)
    # a = mean(logM_med - α^{-1} logλ_med)
    return float(np.mean(np.asarray(ys) - np.asarray(xs)))


def shuffle_null_fn(fn, lam, g, logM_grid, mwd, n_shuf, rng, **kw) -> dict:
    obs = fn(lam, g, logM_grid, mwd, **kw)
    null = np.empty(n_shuf)
    g = np.asarray(g, dtype=np.float64)
    for i in range(n_shuf):
        null[i] = fn(lam, rng.permutation(g), logM_grid, mwd, **kw)
    p = float((np.sum(null >= obs) + 1) / (n_shuf + 1))
    return {"obs": float(obs), "null_mean": float(null.mean()), "delta": float(obs - null.mean()), "p_ge_obs": p}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    ids = sorted(set(saos) & set(gpc))
    logM_grid = np.linspace(3.0, 7.0, 80)

    fits = []
    for sid in ids:
        s = saos[sid]
        lam0 = default_lambda_grid(s.omega, args.n_modes)
        fit = fit_prony_freq_pnf(
            s.omega,
            s.g_prime,
            s.g_double_prime,
            n_modes=args.n_modes,
            fixed_lambda=True,
            lambda_init=lam0,
            seed=args.seed,
        )
        mwd = mwd_on_log_grid(gpc[sid], logM_grid)
        lam = np.asarray(fit.lambda_, dtype=float)
        g = np.asarray(fit.g, dtype=float)
        fits.append(
            {
                "sample": sid,
                "branched": sid in BRANCHED,
                "lam": lam,
                "g": g,
                "mwd": mwd,
                "logM_med": _gpc_logM_med(logM_grid, mwd),
                "loglam_med": _w_loglam_med(lam, g),
            }
        )
        print(f"[b2] fit {sid} rel_log={fit.rel_log:.4f}", flush=True)

    a_global = fit_intercept_a(fits, ALPHA)

    protocols = {}
    # per-sample (legacy)
    rows_ps = []
    for r in fits:
        sh = shuffle_null(r["lam"], r["g"], logM_grid, r["mwd"], ALPHA, args.n_shuffle, rng)
        rows_ps.append({"sample": r["sample"], "branched": r["branched"], **sh, "delta": sh["obs"] - sh["null_mean"]})
    protocols["per_sample"] = rows_ps

    # global intercept
    rows_g = []
    for r in fits:
        sh = shuffle_null_fn(
            tube_corr_intercept,
            r["lam"],
            r["g"],
            logM_grid,
            r["mwd"],
            args.n_shuffle,
            rng,
            alpha=ALPHA,
            a=a_global,
        )
        rows_g.append({"sample": r["sample"], "branched": r["branched"], **sh})
    protocols["global"] = rows_g

    # LOO intercept
    rows_loo = []
    loo_a = {}
    for i, r in enumerate(fits):
        train = [fits[j] for j in range(len(fits)) if j != i]
        a_i = fit_intercept_a(train, ALPHA)
        loo_a[r["sample"]] = a_i
        sh = shuffle_null_fn(
            tube_corr_intercept,
            r["lam"],
            r["g"],
            logM_grid,
            r["mwd"],
            args.n_shuffle,
            rng,
            alpha=ALPHA,
            a=a_i,
        )
        rows_loo.append({"sample": r["sample"], "branched": r["branched"], "a_loo": a_i, **sh})
    protocols["loo"] = rows_loo

    def summarize(rows):
        obs = np.array([r["obs"] for r in rows], float)
        nul = np.array([r["null_mean"] for r in rows], float)
        delta = obs - nul
        # bootstrap mean delta
        boots = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx = rng.integers(0, len(delta), len(delta))
            boots[b] = float(delta[idx].mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        return {
            "mean_r": float(obs.mean()),
            "mean_null": float(nul.mean()),
            "mean_delta": float(delta.mean()),
            "mean_delta_boot_ci95": [float(lo), float(hi)],
            "frac_p_lt_0_05": float(np.mean([r["p_ge_obs"] < 0.05 for r in rows])),
            "n": len(rows),
            "by_sample": {r["sample"]: {"r": r["obs"], "null": r["null_mean"], "delta": r["delta"], "p": r["p_ge_obs"]} for r in rows},
        }

    summary = {
        "alpha": ALPHA,
        "a_global": a_global,
        "loo_a": loo_a,
        "protocols": {k: summarize(v) for k, v in protocols.items()},
        "interpretation": (
            "If only per_sample retains high mean_delta while global/loo collapse, "
            "the claim is shape alignment after target GPC median anchoring, not blind MWD readout."
        ),
    }

    (out / "ANCHORING.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # flat CSV
    lines = ["protocol,sample,branched,r,null_mean,delta,p"]
    for proto, rows in protocols.items():
        for r in rows:
            lines.append(
                f"{proto},{r['sample']},{int(r['branched'])},{r['obs']:.6f},{r['null_mean']:.6f},{r['delta']:.6f},{r['p_ge_obs']:.6f}"
            )
    (out / "by_sample.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        "# Rheology anchoring protocols",
        "",
        f"Global intercept a = **{a_global:.4f}** (α={ALPHA})",
        "",
        "| protocol | mean r | mean null | mean Δ | Δ boot CI95 | frac p<0.05 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for k, s in summary["protocols"].items():
        ci = s["mean_delta_boot_ci95"]
        md.append(
            f"| `{k}` | {s['mean_r']:.3f} | {s['mean_null']:.3f} | {s['mean_delta']:.3f} | "
            f"[{ci[0]:.3f},{ci[1]:.3f}] | {s['frac_p_lt_0_05']:.2f} |"
        )
    md += ["", summary["interpretation"]]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"[b2] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
