#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1: synthetic frequency-domain Prony parameter recovery (PNF vs classical NLS).

Generates SAOS master curves from known {G_k} on a fixed λ library, refits with
PNF and classical Prony NLS, and reports recovery of mode masses / free-λ peaks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
import sys

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.rheo_freq_fit import (
    default_lambda_grid,
    fit_prony_freq_nls,
    fit_prony_freq_pnf,
    score_freq_fit,
)


def _complex_modulus(omega: np.ndarray, lam: np.ndarray, g: np.ndarray, g_inf: float):
    om = np.asarray(omega, float).reshape(-1)
    lam = np.asarray(lam, float).reshape(-1)
    g = np.asarray(g, float).reshape(-1)
    x = om[:, None] * lam[None, :]
    den = 1.0 + x * x
    gp = float(g_inf) + (g[None, :] * (x * x) / den).sum(axis=1)
    gpp = (g[None, :] * x / den).sum(axis=1)
    return gp, gpp


def _mass(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, float).clip(min=0.0)
    s = g.sum()
    if s <= 0:
        return np.full_like(g, 1.0 / max(g.size, 1))
    return g / s


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/rheo/parameter_recovery")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-db", type=float, default=40.0, help="SNR on log moduli; inf disables")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    k = int(args.n_modes)

    omega = np.logspace(-2, 2, 48)
    lam = default_lambda_grid(omega, k)

    rows = []
    for t in range(int(args.n_trials)):
        # Sparse-ish positive mode masses (2–4 alive modes dominate).
        alive = rng.choice(k, size=int(rng.integers(2, 5)), replace=False)
        g_true = np.full(k, 1e-4)
        w = rng.lognormal(mean=0.0, sigma=0.8, size=alive.size)
        g_true[alive] = w / w.sum() * float(rng.uniform(5e4, 5e5))
        g_inf = float(rng.uniform(1e-3, 10.0))

        gp, gpp = _complex_modulus(omega, lam, g_true, g_inf)
        if np.isfinite(args.noise_db):
            sigma = 10 ** (-0.05 * float(args.noise_db))
            gp = gp * np.exp(rng.normal(0.0, sigma, size=gp.shape))
            gpp = gpp * np.exp(rng.normal(0.0, sigma, size=gpp.shape))

        nls = fit_prony_freq_nls(
            omega, gp, gpp, n_modes=k, fixed_lambda=True, lambda_init=lam
        )
        pnf = fit_prony_freq_pnf(
            omega, gp, gpp, n_modes=k, fixed_lambda=True, lambda_init=lam, seed=args.seed + t
        )
        # Free-λ recovery of peak locations (sparse GT).
        pnf_free = fit_prony_freq_pnf(
            omega, gp, gpp, n_modes=min(4, k), fixed_lambda=False, seed=args.seed + 1000 + t
        )

        m_true = _mass(g_true)
        for method, fit in (("classical_prony_nls", nls), ("pnf", pnf)):
            m_hat = _mass(fit.g)
            rows.append(
                {
                    "trial": t,
                    "method": method,
                    "fixed_lambda": True,
                    "rel_log": fit.rel_log,
                    "rel_l2": fit.rel_l2,
                    "mass_l1": float(np.abs(m_hat - m_true).sum()),
                    "mass_spearman": _spearman(m_true, m_hat),
                    "g_cosine": float(
                        np.dot(g_true, fit.g)
                        / (np.linalg.norm(g_true) * np.linalg.norm(fit.g) + 1e-12)
                    ),
                    "success": bool(fit.success),
                }
            )

        # Match free-λ peaks to nearest true alive λ (relative log error).
        true_peaks = lam[alive]
        hat_peaks = np.asarray(pnf_free.lambda_, float)
        hat_peaks = hat_peaks[np.isfinite(hat_peaks) & (hat_peaks > 0)]
        peak_err = []
        for tp in true_peaks:
            if hat_peaks.size == 0 or not np.isfinite(tp) or tp <= 0:
                peak_err.append(float("nan"))
                continue
            peak_err.append(float(np.min(np.abs(np.log10(hat_peaks) - np.log10(tp)))))
        rows.append(
            {
                "trial": t,
                "method": "pnf_free_lambda",
                "fixed_lambda": False,
                "rel_log": pnf_free.rel_log,
                "rel_l2": pnf_free.rel_l2,
                "peak_log10_mae": float(np.mean(peak_err)) if peak_err else float("nan"),
                "n_true_peaks": int(alive.size),
                "success": bool(pnf_free.success),
            }
        )

    def _summ(method: str, key: str) -> dict:
        vals = np.asarray([r[key] for r in rows if r["method"] == method and key in r], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {}
        return {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "n": int(vals.size),
        }

    report = {
        "protocol": {
            "n_trials": int(args.n_trials),
            "n_modes": k,
            "omega": [float(omega.min()), float(omega.max()), int(omega.size)],
            "noise_db": args.noise_db,
            "seed": int(args.seed),
            "claim": (
                "Physics-indexed mode masses are recoverable on synthetic Prony SAOS; "
                "PNF matches classical Prony NLS under the same fixed-λ library."
            ),
        },
        "summary": {
            "pnf_mass_l1": _summ("pnf", "mass_l1"),
            "nls_mass_l1": _summ("classical_prony_nls", "mass_l1"),
            "pnf_mass_spearman": _summ("pnf", "mass_spearman"),
            "nls_mass_spearman": _summ("classical_prony_nls", "mass_spearman"),
            "pnf_g_cosine": _summ("pnf", "g_cosine"),
            "nls_g_cosine": _summ("classical_prony_nls", "g_cosine"),
            "pnf_rel_log": _summ("pnf", "rel_log"),
            "nls_rel_log": _summ("classical_prony_nls", "rel_log"),
            "pnf_free_peak_log10_mae": _summ("pnf_free_lambda", "peak_log10_mae"),
        },
        "rows": rows,
    }
    (out / "RECOVERY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    lines = [
        "# Rheology parameter recovery (synthetic Prony SAOS)",
        "",
        f"- trials={args.n_trials}, modes={k}, noise_db={args.noise_db}",
        f"- PNF mass Spearman mean={s['pnf_mass_spearman'].get('mean', float('nan')):.3f}",
        f"- NLS mass Spearman mean={s['nls_mass_spearman'].get('mean', float('nan')):.3f}",
        f"- PNF mass L1 mean={s['pnf_mass_l1'].get('mean', float('nan')):.3f}",
        f"- NLS mass L1 mean={s['nls_mass_l1'].get('mean', float('nan')):.3f}",
        f"- PNF G cosine mean={s['pnf_g_cosine'].get('mean', float('nan')):.3f}",
        f"- free-λ peak log10 MAE mean={s['pnf_free_peak_log10_mae'].get('mean', float('nan')):.3f}",
        "",
        "Artifact: `RECOVERY.json`.",
    ]
    (out / "RECOVERY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Wrote", out / "RECOVERY.json")


if __name__ == "__main__":
    main()
