#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harden Leeds tube–MWD alignment (α sweep, bootstrap, free-λ, SAOS-bin null).

Does not overwrite spectrum_mwd_mine/; writes outputs/rheo/spectrum_mwd_harden/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from hnf.rheo_freq_fit import default_lambda_grid, fit_prony_freq_pnf
from hnf.rheo_gpc import load_leeds_gpc_all, mwd_on_log_grid
from hnf.rheo_leeds import load_leeds_saos_all

BRANCHED = {"A1PS", "PSA"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--output-dir", default="outputs/rheo/spectrum_mwd_harden")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--n-shuffle", type=int, default=800)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-free-lambda", action="store_true")
    return p.parse_args()


def tube_corr(lam, g, logM_grid, mwd, alpha: float = 3.4) -> float:
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    lam = np.asarray(lam, dtype=np.float64)
    w = g / max(g.sum(), 1e-30)
    mwd = np.asarray(mwd, dtype=np.float64)
    logM_med = float(np.average(logM_grid, weights=np.maximum(mwd, 1e-12)))
    log_lam_med = float(np.average(np.log10(np.maximum(lam, 1e-30)), weights=w))
    logM_of_lam = logM_med + (1.0 / alpha) * (np.log10(np.maximum(lam, 1e-30)) - log_lam_med)
    mwd_mass = np.zeros(len(lam))
    dlog = float(logM_grid[1] - logM_grid[0]) if len(logM_grid) > 1 else 1.0
    for j, lm in enumerate(logM_grid):
        k = int(np.argmin(np.abs(logM_of_lam - lm)))
        mwd_mass[k] += mwd[j] * dlog
    mwd_mass = mwd_mass / max(mwd_mass.sum(), 1e-30)
    if w.std() < 1e-12 or mwd_mass.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(w, mwd_mass)[0, 1])


def saos_bin_proxy(omega, gp, gpp, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Fake 'modes' without Prony: log-ω bins; weight ∝ mean G'' in bin (loss-moduli mass)."""
    omega = np.asarray(omega, dtype=np.float64)
    gpp = np.asarray(gpp, dtype=np.float64)
    order = np.argsort(omega)
    w = omega[order]
    b = np.maximum(gpp[order], 0.0)
    edges = np.logspace(np.log10(w.min()), np.log10(w.max()), n_modes + 1)
    lam = np.empty(n_modes)
    g = np.zeros(n_modes)
    for k in range(n_modes):
        m = (w >= edges[k]) & (w <= edges[k + 1] if k == n_modes - 1 else w < edges[k + 1])
        if not np.any(m):
            lam[k] = 1.0 / float(np.sqrt(edges[k] * edges[k + 1]))
            continue
        lam[k] = 1.0 / float(np.exp(np.mean(np.log(w[m]))))
        g[k] = float(np.mean(b[m]))
    return lam, g


def shuffle_null(lam, g, logM_grid, mwd, alpha, n_shuf, rng) -> dict:
    obs = tube_corr(lam, g, logM_grid, mwd, alpha)
    null = np.empty(n_shuf)
    g = np.asarray(g, dtype=np.float64)
    for i in range(n_shuf):
        null[i] = tube_corr(lam, rng.permutation(g), logM_grid, mwd, alpha)
    p = float((np.sum(null >= obs) + 1) / (n_shuf + 1))
    return {
        "obs": float(obs),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_ge_obs": p,
    }


def bootstrap_mean(obs_list: list[float], n_boot: int, rng) -> dict:
    arr = np.asarray(obs_list, dtype=np.float64)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        boots[i] = float(arr[rng.integers(0, n, n)].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean": float(arr.mean()),
        "ci95": [float(lo), float(hi)],
        "n": n,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    ids = sorted(set(saos) & set(gpc))
    logM_grid = np.linspace(3.0, 7.0, 80)
    alphas = [3.0, 3.4, 3.7]

    fits: dict[str, dict] = {}
    for sid in ids:
        s = saos[sid]
        lam0 = default_lambda_grid(s.omega, args.n_modes)
        fit_fix = fit_prony_freq_pnf(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True, lambda_init=lam0, seed=args.seed,
        )
        pack = {
            "fixed": {"lambda": fit_fix.lambda_.tolist(), "g": fit_fix.g.tolist(), "rel_log": fit_fix.rel_log},
        }
        if not args.skip_free_lambda:
            fit_free = fit_prony_freq_pnf(
                s.omega, s.g_prime, s.g_double_prime,
                n_modes=args.n_modes, fixed_lambda=False, lambda_init=lam0, seed=args.seed,
            )
            pack["free"] = {
                "lambda": fit_free.lambda_.tolist(),
                "g": fit_free.g.tolist(),
                "rel_log": fit_free.rel_log,
            }
        lam_b, g_b = saos_bin_proxy(s.omega, s.g_prime, s.g_double_prime, args.n_modes)
        pack["saos_bin"] = {"lambda": lam_b.tolist(), "g": g_b.tolist()}
        fits[sid] = pack
        print(f"{sid}: fixed rel_log={fit_fix.rel_log:.4f}")

    report: dict = {
        "n_samples": len(ids),
        "samples": ids,
        "alphas": alphas,
        "n_shuffle": args.n_shuffle,
        "n_boot": args.n_boot,
        "by_alpha": {},
        "free_lambda": {},
        "saos_bin_null": {},
        "claim": {},
    }

    for alpha in alphas:
        per = {}
        obs_vals = []
        null_means = []
        for sid in ids:
            mwd = mwd_on_log_grid(gpc[sid], logM_grid)
            lam = fits[sid]["fixed"]["lambda"]
            g = fits[sid]["fixed"]["g"]
            sh = shuffle_null(lam, g, logM_grid, mwd, alpha, args.n_shuffle, rng)
            per[sid] = {**sh, "branched": sid in BRANCHED}
            obs_vals.append(sh["obs"])
            null_means.append(sh["null_mean"])
            print(f"  α={alpha} {sid}: r={sh['obs']:.3f} null={sh['null_mean']:.3f} p={sh['p_ge_obs']:.4f}")
        report["by_alpha"][str(alpha)] = {
            "per_sample": per,
            "obs_boot": bootstrap_mean(obs_vals, args.n_boot, rng),
            "null_boot": bootstrap_mean(null_means, args.n_boot, rng),
            "frac_p_lt_0p05": float(np.mean([per[s]["p_ge_obs"] < 0.05 for s in ids])),
            "delta_obs_minus_null": float(np.mean(obs_vals) - np.mean(null_means)),
        }

    # free-λ at α=3.4
    if not args.skip_free_lambda:
        per_f = {}
        obs_f = []
        null_f = []
        for sid in ids:
            mwd = mwd_on_log_grid(gpc[sid], logM_grid)
            lam = fits[sid]["free"]["lambda"]
            g = fits[sid]["free"]["g"]
            sh = shuffle_null(lam, g, logM_grid, mwd, 3.4, args.n_shuffle, rng)
            per_f[sid] = sh
            obs_f.append(sh["obs"])
            null_f.append(sh["null_mean"])
        report["free_lambda"] = {
            "alpha": 3.4,
            "per_sample": per_f,
            "obs_boot": bootstrap_mean(obs_f, args.n_boot, rng),
            "null_boot": bootstrap_mean(null_f, args.n_boot, rng),
            "delta_obs_minus_null": float(np.mean(obs_f) - np.mean(null_f)),
        }

    # SAOS-bin proxy (no Prony) at α=3.4
    per_b = {}
    obs_b = []
    for sid in ids:
        mwd = mwd_on_log_grid(gpc[sid], logM_grid)
        lam = fits[sid]["saos_bin"]["lambda"]
        g = fits[sid]["saos_bin"]["g"]
        sh = shuffle_null(lam, g, logM_grid, mwd, 3.4, args.n_shuffle, rng)
        per_b[sid] = sh
        obs_b.append(sh["obs"])
    report["saos_bin_null"] = {
        "alpha": 3.4,
        "per_sample": per_b,
        "obs_boot": bootstrap_mean(obs_b, args.n_boot, rng),
        "mean_obs": float(np.mean(obs_b)),
        "note": "Frequency-binned G'' proxy without Prony; should collapse if tube alignment needs Maxwell modes.",
    }

    a34 = report["by_alpha"]["3.4"]
    report["claim"] = {
        "fixed_lambda_alpha_3p4_mean_r": a34["obs_boot"]["mean"],
        "fixed_lambda_alpha_3p4_ci95": a34["obs_boot"]["ci95"],
        "null_mean_r": a34["null_boot"]["mean"],
        "null_ci95": a34["null_boot"]["ci95"],
        "delta": a34["delta_obs_minus_null"],
        "frac_significant": a34["frac_p_lt_0p05"],
        "alpha_sensitivity_means": {
            a: report["by_alpha"][a]["obs_boot"]["mean"] for a in report["by_alpha"]
        },
        "saos_bin_mean_r": report["saos_bin_null"]["mean_obs"],
        "free_lambda_mean_r": (
            report["free_lambda"]["obs_boot"]["mean"] if report["free_lambda"] else None
        ),
    }
    report["fits_meta"] = {
        sid: {
            "fixed_rel_log": fits[sid]["fixed"]["rel_log"],
            "free_rel_log": fits[sid].get("free", {}).get("rel_log"),
        }
        for sid in ids
    }

    (out / "HARDEN.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    c = report["claim"]
    lines = [
        "# Leeds tube–MWD hardening dig",
        "",
        f"**n={len(ids)}** paired SAOS+GPC. Shuffle null on \(G_k\); bootstrap CI on mean r.",
        "",
        "## Primary (fixed-λ, α=3.4)",
        "",
        f"- Observed mean r = **{c['fixed_lambda_alpha_3p4_mean_r']:.3f}** "
        f"(95% CI {c['fixed_lambda_alpha_3p4_ci95'][0]:.3f}–{c['fixed_lambda_alpha_3p4_ci95'][1]:.3f})",
        f"- Shuffle-\(G_k\) null mean r = **{c['null_mean_r']:.3f}** "
        f"(CI {c['null_ci95'][0]:.3f}–{c['null_ci95'][1]:.3f})",
        f"- Δ(obs−null) = **{c['delta']:.3f}**; fraction p<0.05 = **{c['frac_significant']:.2f}**",
        "",
        "## α sensitivity (mean obs r)",
        "",
    ]
    for a, m in c["alpha_sensitivity_means"].items():
        lines.append(f"- α={a}: mean r = **{m:.3f}**")
    lines += [
        "",
        "## Mechanism controls",
        "",
        f"- Free-λ Prony mean r = **{c['free_lambda_mean_r']}** (should stay high if not grid artifact)",
        f"- SAOS-bin proxy (no Prony) mean r = **{c['saos_bin_mean_r']:.3f}** "
        "(should drop if Maxwell modes are required)",
        "",
        "## Nature discipline",
        "",
        "- Hardened positive: tube-aligned mode mass ≫ shuffle null, robust to α∈{3.0,3.4,3.7} if means stay high.",
        "- Still n=9; LOO Mw inversion remains a separate honest negative (see spectrum_mwd_mine).",
        "",
        "Artifact: `HARDEN.json`.",
        "",
    ]
    (out / "HARDEN.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out / "HARDEN.md")
    print("CLAIM:", json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
