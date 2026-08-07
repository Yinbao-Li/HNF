#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare PNF Prony vs diffusion-style memory kernels on Leeds SAOS∩GPC.

Question: does a continuous / fractional diffusion memory improve SAOS fit
and/or tube–MWD alignment relative to discrete PNF modes?

Methods
  - PNF (fixed-λ Prony)
  - diffusion H(λ) continuous spectrum
  - diffusion H(λ) + oscillatory ablation (phase-like)
  - fractional Maxwell (for compare)

Metrics: rel_log fit; tube_corr @ α=3.4; shuffle-G_k null.
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

from hnf.rheo_diffusion_kernel import fit_diffusion_spectrum, fit_fractional_maxwell
from hnf.rheo_freq_fit import fit_prony_freq_pnf
from hnf.rheo_gpc import load_leeds_gpc_all, mwd_on_log_grid
from hnf.rheo_leeds import load_leeds_saos_all
from tools.run_rheo_leeds_tube_harden import shuffle_null, tube_corr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rheo-dir", default="external_data/rheo_leeds_ps/Rheo_Data")
    p.add_argument("--gpc-dir", default="external_data/rheo_leeds_ps/GPC_Data")
    p.add_argument("--output-dir", default="outputs/rheo/diffusion_kernel_vs_prony")
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--n-shuffle", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    saos = {s.sample_id: s for s in load_leeds_saos_all(args.rheo_dir)}
    gpc = {s.sample_id: s for s in load_leeds_gpc_all(args.gpc_dir)}
    ids = sorted(set(saos) & set(gpc))
    logM_grid = np.linspace(3.0, 7.0, 80)

    methods = ("pnf", "diffusion_H", "diffusion_H_osc", "fractional_maxwell")
    rows = []
    by_method: dict[str, list[dict]] = {m: [] for m in methods}

    for sid in ids:
        s = saos[sid]
        mwd = mwd_on_log_grid(gpc[sid], logM_grid)
        print(f"[{sid}] fitting…")

        fits = {
            "pnf": fit_prony_freq_pnf(
                s.omega, s.g_prime, s.g_double_prime, n_modes=args.n_modes, seed=args.seed
            ),
            "diffusion_H": fit_diffusion_spectrum(
                s.omega, s.g_prime, s.g_double_prime, n_modes=args.n_modes, seed=args.seed
            ),
            "diffusion_H_osc": fit_diffusion_spectrum(
                s.omega,
                s.g_prime,
                s.g_double_prime,
                n_modes=args.n_modes,
                oscillatory=True,
                seed=args.seed,
            ),
            "fractional_maxwell": fit_fractional_maxwell(
                s.omega, s.g_prime, s.g_double_prime, n_modes=args.n_modes, seed=args.seed
            ),
        }

        for name, fr in fits.items():
            if name == "pnf":
                lam, g = fr.lambda_, fr.g
                rel_log, rel_l2 = fr.rel_log, fr.rel_l2
            else:
                lam, g = fr.lambda_, fr.g
                rel_log, rel_l2 = fr.rel_log, fr.rel_l2
            sh = shuffle_null(lam, g, logM_grid, mwd, 3.4, args.n_shuffle, rng)
            rec = {
                "sample": sid,
                "method": name,
                "rel_log": float(rel_log),
                "rel_l2": float(rel_l2),
                "tube_r": sh["obs"],
                "null_mean": sh["null_mean"],
                "delta": float(sh["obs"] - sh["null_mean"]),
                "p_ge_obs": sh["p_ge_obs"],
            }
            rows.append(rec)
            by_method[name].append(rec)
            print(
                f"  {name:20s} rel_log={rel_log:.4f}  tube_r={sh['obs']:.3f}  "
                f"null={sh['null_mean']:.3f}  Δ={sh['obs']-sh['null_mean']:.3f}"
            )

    summary = {"n": len(ids), "samples": ids, "methods": {}}
    for m, lst in by_method.items():
        summary["methods"][m] = {
            "mean_rel_log": float(np.mean([r["rel_log"] for r in lst])),
            "mean_tube_r": float(np.mean([r["tube_r"] for r in lst])),
            "mean_null": float(np.mean([r["null_mean"] for r in lst])),
            "mean_delta": float(np.mean([r["delta"] for r in lst])),
            "frac_p_lt_0p05": float(np.mean([r["p_ge_obs"] < 0.05 for r in lst])),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_sample.json").write_text(json.dumps(rows, indent=2))

    # compact CSV
    lines = ["sample,method,rel_log,rel_l2,tube_r,null_mean,delta,p_ge_obs"]
    for r in rows:
        lines.append(
            f"{r['sample']},{r['method']},{r['rel_log']:.6f},{r['rel_l2']:.6f},"
            f"{r['tube_r']:.6f},{r['null_mean']:.6f},{r['delta']:.6f},{r['p_ge_obs']:.6f}"
        )
    (out / "per_sample.csv").write_text("\n".join(lines) + "\n")

    sm = summary["methods"]
    md = [
        "# Diffusion memory kernel vs PNF (Leeds SAOS∩GPC n=9)",
        "",
        "Question: does a continuous diffusion-style H(λ) (or fractional Maxwell)",
        "beat discrete **PNF** on SAOS fit and/or tube–MWD alignment?",
        "",
        "| method | mean rel_log ↓ | mean tube r ↑ | mean Δ(obs−null) ↑ | frac p<0.05 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in methods:
        s = sm[m]
        md.append(
            f"| `{m}` | {s['mean_rel_log']:.4f} | {s['mean_tube_r']:.3f} | "
            f"{s['mean_delta']:.3f} | {s['frac_p_lt_0p05']:.2f} |"
        )
    md += [
        "",
        "## Reading",
        "- Better **rel_log** alone ≠ better discovery: continuous spectra can fit",
        "  G′,G″ while washing out readable mode mass.",
        "- **diffusion_H_osc** is the EEG-style phase ablation: oscillating H(λ)",
        "  should hurt tube alignment if mass placement (not mere smoothness) matters.",
        "- Fractional Maxwell is a low-parameter diffusion/fractional comparator,",
        "  not a PNF replacement.",
        "",
        f"Artifacts: `{out}/`",
    ]
    (out / "BOARD.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n[board] → {out / 'BOARD.md'}")


if __name__ == "__main__":
    main()
