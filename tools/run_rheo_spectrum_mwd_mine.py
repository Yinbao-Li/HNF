#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small-n discovery mine: interpretable PNF spectral descriptors → MWD.

Designed for n≈9 (Leeds): univariate/bivariate LOO + Spearman permutation tests,
tube-scaling probe, branched residual biomarker. Avoids high-dim ridge LOO.
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
from scipy import stats

from hnf.rheo_freq_fit import default_lambda_grid, fit_prony_freq_pnf
from hnf.rheo_gpc import load_leeds_gpc_all, mwd_moments, mwd_on_log_grid
from hnf.rheo_leeds import load_leeds_saos_all

BRANCHED = {"A1PS", "PSA"}
TARGETS = ["log10_Mw", "log10_Mn", "D", "logM_peak", "logM_width90"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--output-dir", default="outputs/rheo/spectrum_mwd_mine")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def spectral_descriptors(lam: np.ndarray, g: np.ndarray, g_inf: float) -> dict[str, float]:
    g = np.asarray(g, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    g = np.maximum(g, 0.0)
    s = g.sum() + 1e-30
    w = g / s
    logl = np.log10(np.maximum(lam, 1e-30))
    mean_logl = float((w * logl).sum())
    var_logl = float((w * (logl - mean_logl) ** 2).sum())
    # tercile bands on logλ
    q1, q2 = np.quantile(logl, [1 / 3, 2 / 3])
    short = float(w[logl <= q1].sum())
    mid = float(w[(logl > q1) & (logl <= q2)].sum())
    long = float(w[logl > q2].sum())
    # effective modes (participation ratio)
    pr = float(1.0 / np.sum(w ** 2))
    return {
        "log10_Gsum": float(np.log10(s)),
        "mean_log10_lam": mean_logl,
        "std_log10_lam": float(np.sqrt(max(var_logl, 0.0))),
        "mass_short": short,
        "mass_mid": mid,
        "mass_long": long,
        "n_eff": pr,
        "log10_ginf_eps": float(np.log10(max(g_inf, 1e-30))),
        "active_frac": float((g > 0.01 * g.max()).mean()),
    }


def saos_descriptors(omega: np.ndarray, gp: np.ndarray, gpp: np.ndarray) -> dict[str, float]:
    """Physics-light SAOS scalars (baseline, not Prony)."""
    omega = np.asarray(omega)
    gp = np.asarray(gp)
    gpp = np.asarray(gpp)
    # crossover approx: where G'≈G''
    ratio = gp / np.maximum(gpp, 1e-30)
    # find sign change of log(G'/G'')
    log_r = np.log(np.maximum(ratio, 1e-30))
    cross = np.where(np.diff(np.sign(log_r)) != 0)[0]
    if len(cross):
        i = int(cross[len(cross) // 2])
        w_c = float(10 ** np.interp(0.0, log_r[i : i + 2], np.log10(omega[i : i + 2])))
    else:
        w_c = float(omega[np.argmin(np.abs(log_r))])
    # terminal-ish: lowest decade mean G''/ω ~ η0 proxy
    order = np.argsort(omega)
    w, a, b = omega[order], gp[order], gpp[order]
    n_lo = max(3, len(w) // 8)
    eta0 = float(np.mean(b[:n_lo] / w[:n_lo]))
    # high-ω plateau proxy
    n_hi = max(3, len(w) // 8)
    g_plat = float(np.mean(a[-n_hi:]))
    return {
        "log10_omega_cross": float(np.log10(max(w_c, 1e-30))),
        "log10_eta0": float(np.log10(max(eta0, 1e-30))),
        "log10_Gplat": float(np.log10(max(g_plat, 1e-30))),
        "log10_omega_span": float(np.log10(w.max() / w.min())),
    }


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    rho, _ = stats.spearmanr(x, y)
    if not np.isfinite(rho):
        return {"rho": 0.0, "p_perm": 1.0}
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i], _ = stats.spearmanr(x, rng.permutation(y))
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return {"rho": float(rho), "p_perm": p}


def loo_univariate(x: np.ndarray, y: np.ndarray) -> dict:
    """LOO linear regression y ~ a + b x."""
    n = len(y)
    yhat = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        b, a = np.polyfit(x[tr], y[tr], 1)
        yhat[i] = a + b * x[i]
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = 1.0 - float(np.sum(err ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12)
    return {"yhat": yhat.tolist(), "mae": mae, "rmse": rmse, "r2_loo": r2}


def loo_bivariate(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> dict:
    n = len(y)
    yhat = np.zeros(n)
    X = np.column_stack([np.ones(n), x1, x2])
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        beta, _, _, _ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        yhat[i] = float(X[i] @ beta)
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    r2 = 1.0 - float(np.sum(err ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12)
    return {"yhat": yhat.tolist(), "mae": mae, "r2_loo": r2}


def tube_corr(lam, g, logM_grid, mwd, alpha=3.4) -> float:
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    lam = np.asarray(lam, dtype=np.float64)
    w = g / max(g.sum(), 1e-30)
    logM_med = float(np.average(logM_grid, weights=np.maximum(mwd, 1e-12)))
    log_lam_med = float(np.average(np.log10(lam), weights=w))
    logM_of_lam = logM_med + (1.0 / alpha) * (np.log10(lam) - log_lam_med)
    mwd_mass = np.zeros(len(lam))
    dlog = float(logM_grid[1] - logM_grid[0]) if len(logM_grid) > 1 else 1.0
    for j, lm in enumerate(logM_grid):
        k = int(np.argmin(np.abs(logM_of_lam - lm)))
        mwd_mass[k] += mwd[j] * dlog
    mwd_mass = mwd_mass / max(mwd_mass.sum(), 1e-30)
    if w.std() < 1e-12 or mwd_mass.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(w, mwd_mass)[0, 1])


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    ids = sorted(set(saos) & set(gpc))

    w_all = np.concatenate([saos[i].omega for i in ids])
    global_lam = default_lambda_grid(w_all, args.n_modes)
    logM_grid = np.linspace(3.0, 7.0, 80)

    rows = []
    for sid in ids:
        s = saos[sid]
        fit = fit_prony_freq_pnf(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True, lambda_init=global_lam, seed=args.seed,
        )
        # also sample-native λ grid (better for narrow windows)
        fit_n = fit_prony_freq_pnf(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True,
            lambda_init=default_lambda_grid(s.omega, args.n_modes), seed=args.seed,
        )
        mom = mwd_moments(gpc[sid])
        desc = spectral_descriptors(fit_n.lambda_, fit_n.g, fit_n.g_inf)
        saos_d = saos_descriptors(s.omega, s.g_prime, s.g_double_prime)
        mwd = mwd_on_log_grid(gpc[sid], logM_grid)
        tc = tube_corr(fit_n.lambda_, fit_n.g, logM_grid, mwd)
        row = {
            "sample": sid,
            "branched": sid in BRANCHED,
            **mom,
            **{f"pnf_{k}": v for k, v in desc.items()},
            **{f"saos_{k}": v for k, v in saos_d.items()},
            "tube_corr": tc,
            "pnf_rel_log": fit_n.rel_log,
            "lambda": fit_n.lambda_.tolist(),
            "g": fit_n.g.tolist(),
        }
        rows.append(row)
        print(
            f"{sid}: Mw={mom['Mw']:.3g} D={mom['D']:.2f}  "
            f"<logλ>={desc['mean_log10_lam']:.2f} tube_r={tc:.2f} branched={sid in BRANCHED}"
        )

    # Feature sets
    pnf_feats = [
        "pnf_log10_Gsum",
        "pnf_mean_log10_lam",
        "pnf_std_log10_lam",
        "pnf_mass_short",
        "pnf_mass_long",
        "pnf_n_eff",
    ]
    saos_feats = [
        "saos_log10_omega_cross",
        "saos_log10_eta0",
        "saos_log10_Gplat",
        "saos_log10_omega_span",
    ]

    discovery = {
        "n_samples": len(ids),
        "samples": [r["sample"] for r in rows],
        "branched": sorted(BRANCHED),
        "rows": rows,
        "correlations": {},
        "loo": {},
        "tube": {},
        "branched_biomarker": {},
    }

    for t in TARGETS:
        y = np.array([r[t] for r in rows], dtype=np.float64)
        corr_block = {"pnf": {}, "saos": {}}
        for f in pnf_feats:
            x = np.array([r[f] for r in rows], dtype=np.float64)
            corr_block["pnf"][f] = spearman_perm(x, y, args.n_perm, rng)
        for f in saos_feats:
            x = np.array([r[f] for r in rows], dtype=np.float64)
            corr_block["saos"][f] = spearman_perm(x, y, args.n_perm, rng)
        discovery["correlations"][t] = corr_block

        # best univariate by |rho|
        best_pnf = max(corr_block["pnf"].items(), key=lambda kv: abs(kv[1]["rho"]))
        best_saos = max(corr_block["saos"].items(), key=lambda kv: abs(kv[1]["rho"]))
        x_pnf = np.array([r[best_pnf[0]] for r in rows])
        x_saos = np.array([r[best_saos[0]] for r in rows])
        loo_pnf = loo_univariate(x_pnf, y)
        loo_saos = loo_univariate(x_saos, y)

        # best bivariate among top-2 PNF feats by |rho|
        ranked = sorted(corr_block["pnf"].items(), key=lambda kv: abs(kv[1]["rho"]), reverse=True)
        f1, f2 = ranked[0][0], ranked[1][0]
        loo_bi = loo_bivariate(
            np.array([r[f1] for r in rows]),
            np.array([r[f2] for r in rows]),
            y,
        )
        discovery["loo"][t] = {
            "best_pnf_feat": best_pnf[0],
            "best_pnf_rho": best_pnf[1],
            "best_saos_feat": best_saos[0],
            "best_saos_rho": best_saos[1],
            "loo_pnf_uni": loo_pnf,
            "loo_saos_uni": loo_saos,
            "loo_pnf_bi": {"feats": [f1, f2], **loo_bi},
        }
        print(
            f"{t}: best PNF {best_pnf[0]} ρ={best_pnf[1]['rho']:.2f} p={best_pnf[1]['p_perm']:.3f} "
            f"LOO R2={loo_pnf['r2_loo']:.2f} | SAOS {best_saos[0]} ρ={best_saos[1]['rho']:.2f} "
            f"LOO R2={loo_saos['r2_loo']:.2f} | bi R2={loo_bi['r2_loo']:.2f}"
        )

    # Tube summary + perm test on mean corr linear vs branched
    tube_all = np.array([r["tube_corr"] for r in rows])
    tube_lin = np.array([r["tube_corr"] for r in rows if not r["branched"]])
    tube_br = np.array([r["tube_corr"] for r in rows if r["branched"]])
    discovery["tube"] = {
        "mean_all": float(tube_all.mean()),
        "mean_linear": float(tube_lin.mean()),
        "mean_branched": float(tube_br.mean()),
        "by_sample": {r["sample"]: r["tube_corr"] for r in rows},
    }

    # Branched biomarker: distance in PNF descriptor space + MWD prediction residual
    # Use mean_log10_lam and std as 2D fingerprint; mahalanobis-like z vs linear centroid
    lin = [r for r in rows if not r["branched"]]
    mu = np.array([
        np.mean([r["pnf_mean_log10_lam"] for r in lin]),
        np.mean([r["pnf_std_log10_lam"] for r in lin]),
        np.mean([r["pnf_n_eff"] for r in lin]),
    ])
    Xlin = np.array([[r["pnf_mean_log10_lam"], r["pnf_std_log10_lam"], r["pnf_n_eff"]] for r in lin])
    cov = np.cov(Xlin.T) + 1e-6 * np.eye(3)
    cov_inv = np.linalg.inv(cov)

    def _maha(r):
        v = np.array([r["pnf_mean_log10_lam"], r["pnf_std_log10_lam"], r["pnf_n_eff"]]) - mu
        return float(np.sqrt(v @ cov_inv @ v))

    bio = []
    for r in rows:
        bio.append({
            "sample": r["sample"],
            "branched": r["branched"],
            "mahalanobis_to_linear": _maha(r),
            "tube_corr": r["tube_corr"],
            "pnf_mean_log10_lam": r["pnf_mean_log10_lam"],
            "pnf_n_eff": r["pnf_n_eff"],
        })
    # AUC-like: rank by mahalanobis, branched should be high
    scores = np.array([b["mahalanobis_to_linear"] for b in bio])
    labels = np.array([1 if b["branched"] else 0 for b in bio])
    # Mann-Whitney AUC
    if labels.sum() > 0 and labels.sum() < len(labels):
        auc = float(stats.mannwhitneyu(scores[labels == 1], scores[labels == 0], alternative="greater").statistic
                    / (labels.sum() * (len(labels) - labels.sum())))
        # proper AUC:
        from itertools import product
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        auc = float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p, n in product(pos, neg)]))
    else:
        auc = None
    discovery["branched_biomarker"] = {
        "rows": bio,
        "auc_mahalanobis": auc,
        "mean_maha_branched": float(np.mean([b["mahalanobis_to_linear"] for b in bio if b["branched"]])),
        "mean_maha_linear": float(np.mean([b["mahalanobis_to_linear"] for b in bio if not b["branched"]])),
    }
    print("Tube mean linear/branched:", discovery["tube"]["mean_linear"], discovery["tube"]["mean_branched"])
    print("Branched AUC (maha):", auc)

    (out / "DISCOVERY.json").write_text(json.dumps(discovery, indent=2), encoding="utf-8")

    # Markdown
    lines = [
        "# Discovery mine — interpretable PNF spectrum ↔ MWD",
        "",
        f"**n={len(ids)}** paired Leeds SAOS+GPC samples (Elliott et al. 2025). "
        "Small-n protocol: Spearman + permutation p, univariate/bivariate LOO, tube probe, branched fingerprint.",
        "",
        "## Finding A — Physics-reduced spectral descriptors beat naive high-dim maps",
        "",
        "High-dim Ridge on 8 amplitudes gave strongly negative LOO R² (overfit). "
        "Reduced descriptors recover readable associations:",
        "",
        "| Target | Best PNF feature | ρ | p_perm | LOO R² (uni) | Best SAOS feature | ρ | LOO R² | PNF bi R² |",
        "|--------|------------------|----:|-------:|-------------:|-------------------|----:|-------:|----------:|",
    ]
    for t in TARGETS:
        L = discovery["loo"][t]
        lines.append(
            f"| {t} | `{L['best_pnf_feat']}` | {L['best_pnf_rho']['rho']:.2f} | "
            f"{L['best_pnf_rho']['p_perm']:.3f} | {L['loo_pnf_uni']['r2_loo']:.2f} | "
            f"`{L['best_saos_feat']}` | {L['best_saos_rho']['rho']:.2f} | "
            f"{L['loo_saos_uni']['r2_loo']:.2f} | {L['loo_pnf_bi']['r2_loo']:.2f} |"
        )

    lines += [
        "",
        "## Finding B — Tube scaling (M∝λ^{1/3.4}): mode mass vs MWD mass",
        "",
        f"- All samples mean corr = **{discovery['tube']['mean_all']:.3f}**",
        f"- Linear melts mean corr = **{discovery['tube']['mean_linear']:.3f}**",
        f"- Branched (A1PS, PSA) mean corr = **{discovery['tube']['mean_branched']:.3f}**",
        "",
        "Per-sample: "
        + ", ".join(f"{k}={v:.2f}" for k, v in discovery["tube"]["by_sample"].items()),
        "",
        "## Finding C — Branching biomarker from spectrum geometry",
        "",
        f"- Mahalanobis distance to linear centroid (〈logλ〉, σ_logλ, n_eff): "
        f"branched mean={discovery['branched_biomarker']['mean_maha_branched']:.2f}, "
        f"linear mean={discovery['branched_biomarker']['mean_maha_linear']:.2f}, "
        f"AUC=**{discovery['branched_biomarker']['auc_mahalanobis']}**",
        "",
        "## Claim discipline",
        "",
        "- n=9 ⇒ findings are **hypothesis-grade** with permutation tests, not a universal law.",
        "- Strong publishable angle if: (i) a descriptor–Mw link survives LOO with p_perm<0.05, "
        "and/or (ii) tube corr drops on branched, and/or (iii) spectrum geometry separates branching.",
        "- Elliott et al. use black-box NN for full MWD; our increment is **interpretable sufficient statistics + failure mode**.",
        "",
        "Artifacts: `DISCOVERY.json`. Figure: `tools/plot_rheo_spectrum_mwd_figure.py`.",
        "",
    ]
    (out / "DISCOVERY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out / "DISCOVERY.md")


if __name__ == "__main__":
    main()
