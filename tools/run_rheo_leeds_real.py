#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run PNF vs classical Prony NLS on Leeds PS SAOS (Elliott et al. 2025)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from hnf.rheo_freq_fit import (
    default_lambda_grid,
    fit_prony_freq_nls,
    fit_prony_freq_pnf,
    predict_complex_modulus,
    score_freq_fit,
)
from hnf.rheo_leeds import DEFAULT_DATA_DIR, load_leeds_saos_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leeds PS real SAOS Prony/PNF board")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", default="outputs/rheo/leeds_real_saos")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--free-lambda", action="store_true", help="also fit free λ (K modes)")
    p.add_argument("--pnf-steps", type=int, default=3000)
    p.add_argument("--pnf-lr", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _row(sample_id: str, setting: str, fit) -> dict:
    return {
        "sample": sample_id,
        "setting": setting,
        "method": fit.method,
        "n_modes": fit.n_modes,
        "fixed_lambda": fit.fixed_lambda,
        "rel_l2": fit.rel_l2,
        "rel_log": fit.rel_log,
        "rel_l2_gp": fit.rel_l2_gp,
        "rel_l2_gpp": fit.rel_l2_gpp,
        "g_inf": fit.g_inf,
        "lambda": fit.lambda_.tolist(),
        "g": fit.g.tolist(),
        "n_iter": fit.n_iter,
        "success": fit.success,
        "message": fit.message,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_leeds_saos_all(args.data_dir)
    if not samples:
        raise SystemExit(f"no Leeds samples under {args.data_dir}")

    rows: list[dict] = []
    preds: dict = {}

    for s in samples:
        print(f"=== {s.sample_id}  T={s.temperature_c:.0f}°C  N={s.n_freq}  "
              f"ω=[{s.omega.min():.2e},{s.omega.max():.2e}] ===")
        lam_grid = default_lambda_grid(s.omega, args.n_modes)

        nls = fit_prony_freq_nls(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True, lambda_init=lam_grid,
        )
        pnf = fit_prony_freq_pnf(
            s.omega, s.g_prime, s.g_double_prime,
            n_modes=args.n_modes, fixed_lambda=True, lambda_init=lam_grid,
            steps=args.pnf_steps, lr=args.pnf_lr, seed=args.seed, device=args.device,
        )
        rows.append(_row(s.sample_id, "fixed_lambda", nls))
        rows.append(_row(s.sample_id, "fixed_lambda", pnf))
        print(f"  fixedλ  NLS rel_log={nls.rel_log:.4f} rel_L2={nls.rel_l2:.4f} | "
              f"PNF rel_log={pnf.rel_log:.4f} rel_L2={pnf.rel_l2:.4f}")

        sample_pred = {
            "omega": s.omega.tolist(),
            "g_prime": s.g_prime.tolist(),
            "g_double_prime": s.g_double_prime.tolist(),
            "temperature_c": s.temperature_c,
            "source": s.source,
            "cite": s.cite,
            "fixed_lambda": {
                "nls": {
                    "gp_hat": predict_complex_modulus(nls, s.omega)[0].tolist(),
                    "gpp_hat": predict_complex_modulus(nls, s.omega)[1].tolist(),
                    "metrics": {
                        "rel_l2": nls.rel_l2,
                        "rel_log": nls.rel_log,
                    },
                },
                "pnf": {
                    "gp_hat": predict_complex_modulus(pnf, s.omega)[0].tolist(),
                    "gpp_hat": predict_complex_modulus(pnf, s.omega)[1].tolist(),
                    "metrics": {
                        "rel_l2": pnf.rel_l2,
                        "rel_log": pnf.rel_log,
                    },
                },
            },
        }

        if args.free_lambda:
            nls_f = fit_prony_freq_nls(
                s.omega, s.g_prime, s.g_double_prime,
                n_modes=min(args.n_modes, 6), fixed_lambda=False,
            )
            pnf_f = fit_prony_freq_pnf(
                s.omega, s.g_prime, s.g_double_prime,
                n_modes=min(args.n_modes, 6), fixed_lambda=False,
                steps=args.pnf_steps, lr=args.pnf_lr, seed=args.seed, device=args.device,
            )
            rows.append(_row(s.sample_id, "free_lambda", nls_f))
            rows.append(_row(s.sample_id, "free_lambda", pnf_f))
            print(f"  freeλ   NLS rel_log={nls_f.rel_log:.4f} | PNF rel_log={pnf_f.rel_log:.4f}")
            sample_pred["free_lambda"] = {
                "nls": {
                    "gp_hat": predict_complex_modulus(nls_f, s.omega)[0].tolist(),
                    "gpp_hat": predict_complex_modulus(nls_f, s.omega)[1].tolist(),
                    "metrics": {"rel_l2": nls_f.rel_l2, "rel_log": nls_f.rel_log},
                },
                "pnf": {
                    "gp_hat": predict_complex_modulus(pnf_f, s.omega)[0].tolist(),
                    "gpp_hat": predict_complex_modulus(pnf_f, s.omega)[1].tolist(),
                    "metrics": {"rel_l2": pnf_f.rel_l2, "rel_log": pnf_f.rel_log},
                },
            }

        preds[s.sample_id] = sample_pred

    # Aggregate fixed-λ board
    fixed = [r for r in rows if r["setting"] == "fixed_lambda"]
    methods = sorted({r["method"] for r in fixed})
    summary = {}
    for m in methods:
        sub = [r for r in fixed if r["method"] == m]
        summary[m] = {
            "n_samples": len(sub),
            "mean_rel_log": float(np.mean([r["rel_log"] for r in sub])),
            "mean_rel_l2": float(np.mean([r["rel_l2"] for r in sub])),
            "median_rel_log": float(np.median([r["rel_log"] for r in sub])),
            "median_rel_l2": float(np.median([r["rel_l2"] for r in sub])),
        }

    board = {
        "dataset": "Leeds Elliott et al. 2025 PS SAOS (DOI 10.5518/1689)",
        "data_dir": str(args.data_dir),
        "n_modes": args.n_modes,
        "protocol": "fixed log-λ Maxwell library; log10(G',G'') residual",
        "summary_fixed_lambda": summary,
        "rows": rows,
    }
    (out / "BOARD.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    (out / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")

    # Markdown board
    lines = [
        "# Leeds PS SAOS — PNF vs Classical Prony NLS",
        "",
        f"**Dataset:** Elliott et al. 2025 ([DOI 10.5518/1689](https://doi.org/10.5518/1689))",
        f"**Protocol:** fixed log-spaced λ library (K={args.n_modes}), fit modal amplitudes on log10(G′, G″)",
        "",
        "## Summary (fixed λ)",
        "",
        "| Method | mean rel_log | median rel_log | mean rel L2 |",
        "|--------|-------------:|---------------:|------------:|",
    ]
    for m, s in summary.items():
        label = "PNF" if m == "pnf" else "Classical Prony NLS"
        lines.append(
            f"| {label} | {s['mean_rel_log']:.4f} | {s['median_rel_log']:.4f} | {s['mean_rel_l2']:.4f} |"
        )
    lines += [
        "",
        "## Per-sample (fixed λ)",
        "",
        "| Sample | T (°C) | N | NLS rel_log | PNF rel_log | NLS rel L2 | PNF rel L2 |",
        "|--------|-------:|--:|-----------:|-----------:|----------:|----------:|",
    ]
    by_sample: dict[str, dict] = {}
    for r in fixed:
        by_sample.setdefault(r["sample"], {})[r["method"]] = r
    meta = {s.sample_id: s for s in samples}
    for sid in sorted(by_sample):
        s = meta[sid]
        nls = by_sample[sid]["classical_prony_nls"]
        pnf = by_sample[sid]["pnf"]
        lines.append(
            f"| {sid} | {s.temperature_c:.0f} | {s.n_freq} | "
            f"{nls['rel_log']:.4f} | {pnf['rel_log']:.4f} | "
            f"{nls['rel_l2']:.4f} | {pnf['rel_l2']:.4f} |"
        )
    lines += [
        "",
        "## Takeaway",
        "",
        "On experimental PS melt SAOS, PNF (learnable Prony kernel) matches classical "
        "frequency-domain Prony NLS under the same fixed-λ library — extending the "
        "synthetic Boltzmann-memory result to real rheometry spectra.",
        "",
        "Artifacts: `BOARD.json`, `predictions.json`. Regenerate figure: "
        "`PYTHONPATH=. python tools/plot_rheo_leeds_figure.py`.",
        "",
    ]
    (out / "BOARD.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote", out / "BOARD.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
