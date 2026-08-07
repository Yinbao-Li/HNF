#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UMN star→bottlebrush SAOS board (DOI 10.13020/y7as-3w53).

Chemistry / topology diversity only — NOT tube–MWD (SEC is chromatogram, not w(M)).
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

from hnf.rheo_freq_fit import default_lambda_grid, fit_prony_freq_nls, fit_prony_freq_pnf
from hnf.rheo_umn import DOI, T_REF_C, load_umn_saos_all

# Zografos et al.: star→bottlebrush transition near Nbb ≈ 50–69
STAR_MAX = 50
BOTTLE_MIN = 70


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/rheo/umn_bottlebrush")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--include-suspect", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def regime(nbb: int) -> str:
    if nbb <= STAR_MAX:
        return "star-like"
    if nbb >= BOTTLE_MIN:
        return "bottlebrush"
    return "transition"


def spectral_scalars(lam, g) -> dict[str, float]:
    g = np.maximum(np.asarray(g, dtype=np.float64), 0.0)
    lam = np.asarray(lam, dtype=np.float64)
    w = g / max(g.sum(), 1e-30)
    logl = np.log10(np.maximum(lam, 1e-30))
    mean = float((w * logl).sum())
    var = float((w * (logl - mean) ** 2).sum())
    return {
        "mean_log10_lam": mean,
        "std_log10_lam": float(np.sqrt(max(var, 0.0))),
        "n_eff": float(1.0 / np.sum(w ** 2)),
        "log10_Gsum": float(np.log10(max(g.sum(), 1e-30))),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = load_umn_saos_all()
    if not args.include_suspect:
        samples = [s for s in samples if s.quality_ok]

    rows = []
    for s in samples:
        lam0 = default_lambda_grid(s.omega, args.n_modes)
        nls = fit_prony_freq_nls(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True, lambda_init=lam0,
        )
        pnf = fit_prony_freq_pnf(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True, lambda_init=lam0, seed=args.seed,
        )
        desc = spectral_scalars(pnf.lambda_, pnf.g)
        row = {
            "sample_id": s.sample_id,
            "nbb": s.nbb,
            "nsc": s.nsc,
            "regime": regime(s.nbb),
            "quality_ok": s.quality_ok,
            "quality_note": s.quality_note,
            "n_freq": s.n_freq,
            "T_ref_C": T_REF_C,
            "nls_rel_log": nls.rel_log,
            "pnf_rel_log": pnf.rel_log,
            "abs_diff_rel_log": abs(pnf.rel_log - nls.rel_log),
            **{f"pnf_{k}": v for k, v in desc.items()},
            "lambda": pnf.lambda_.tolist(),
            "g": pnf.g.tolist(),
        }
        rows.append(row)
        print(
            f"{s.sample_id} [{row['regime']}]: NLS={nls.rel_log:.4f} PNF={pnf.rel_log:.4f} "
            f"<logλ>={desc['mean_log10_lam']:.2f}"
        )

    nls_m = float(np.mean([r["nls_rel_log"] for r in rows]))
    pnf_m = float(np.mean([r["pnf_rel_log"] for r in rows]))
    # Spearman Nbb vs mean_log10_lam
    from scipy import stats

    nbb = np.array([r["nbb"] for r in rows], dtype=float)
    mean_l = np.array([r["pnf_mean_log10_lam"] for r in rows], dtype=float)
    rho, pval = stats.spearmanr(nbb, mean_l)

    board = {
        "doi": DOI,
        "cite": "Zografos et al.; DRUM 10.13020/y7as-3w53",
        "n_samples": len(rows),
        "excluded_suspect": not args.include_suspect,
        "star_max_nbb": STAR_MAX,
        "bottle_min_nbb": BOTTLE_MIN,
        "mean_nls_rel_log": nls_m,
        "mean_pnf_rel_log": pnf_m,
        "pnf_equals_nls": abs(pnf_m - nls_m) < 1e-3,
        "spearman_nbb_vs_mean_loglam": {"rho": float(rho), "p": float(pval)},
        "rows": rows,
        "claim_limit": (
            "PLA graft SAOS diversity + PNF≡NLS transfer; "
            "NOT calibrated MWD / tube–MWD alignment (SEC is time–dRI)."
        ),
    }
    (out / "BOARD.json").write_text(json.dumps(board, indent=2), encoding="utf-8")

    lines = [
        "# UMN star→bottlebrush SAOS board",
        "",
        f"DOI [{DOI}](https://doi.org/{DOI}). T_ref = {T_REF_C} °C. "
        f"n={len(rows)} quality-ok master curves (Nbb=210 excluded by default: corrupt G').",
        "",
        f"- Mean rel_log: NLS=**{nls_m:.4f}**, PNF=**{pnf_m:.4f}** → PNF≡NLS on PLA grafts.",
        f"- Spearman(Nbb, 〈log₁₀ λ〉): ρ=**{rho:.2f}**, p=**{pval:.3g}**",
        f"- Regime labels: star-like Nbb≤{STAR_MAX}, bottlebrush Nbb≥{BOTTLE_MIN} (paper transition ~50–69).",
        "",
        "## Claim discipline",
        "",
        "- Use for **topology/chemistry diversity** of readable spectra.",
        "- Do **not** claim tube–MWD here (no calibrated w(M)).",
        "",
        "Artifact: `BOARD.json`.",
        "",
    ]
    (out / "BOARD.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out / "BOARD.md")


if __name__ == "__main__":
    main()
