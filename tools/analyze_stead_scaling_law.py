#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit STEAD scaling laws and effective-DoF / sample-efficiency summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

C = {"hnf": "#c45c26", "phasenet": "#2c6e8a", "eqtransformer": "#5a7d4e"}
LABEL = {"hnf": "HNF", "phasenet": "PhaseNet", "eqtransformer": "EQTransformer"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/stead_scaling_law")
    p.add_argument("--out-fig", default="docs/figures/stead/stead_journal_scaling_law")
    return p.parse_args()


def power_err(n, a, alpha, c):
    return c + a * np.power(n, -alpha)


def fit_scaling(n: np.ndarray, y: np.ndarray) -> dict:
    """Fit y(N)=c + a N^{-α} with y = 1-F1 or MAE."""
    n = np.asarray(n, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(n) & np.isfinite(y) & (n > 0) & (y >= 0)
    n, y = n[mask], y[mask]
    if len(n) < 3:
        return {"ok": False, "n": n.tolist(), "y": y.tolist()}
    # init
    c0 = max(float(y.min()) * 0.5, 1e-4)
    a0 = max(float(y.max() - c0), 1e-3)
    try:
        popt, pcov = curve_fit(
            power_err,
            n,
            y,
            p0=[a0, 0.3, c0],
            bounds=([1e-8, 0.01, 0.0], [10.0, 3.0, 1.0]),
            maxfev=20000,
        )
        a, alpha, c = map(float, popt)
        yhat = power_err(n, a, alpha, c)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        return {
            "ok": True,
            "a": a,
            "alpha": alpha,
            "c": c,
            "r2": 1.0 - ss_res / ss_tot,
            "n": n.tolist(),
            "y": y.tolist(),
            "yhat": yhat.tolist(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "n": n.tolist(), "y": y.tolist()}


def n_to_reach(fit: dict, target_err: float) -> float | None:
    """Invert c + a N^{-α} = target_err → N."""
    if not fit.get("ok"):
        return None
    a, alpha, c = fit["a"], fit["alpha"], fit["c"]
    if target_err <= c + 1e-9:
        return None
    return float((a / (target_err - c)) ** (1.0 / alpha))


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.glob("*/test_metrics.json")):
        m = json.loads(p.read_text())
        model = m.get("model") or p.parent.name.split("_N")[0]
        n_ev = m.get("n_event_train")
        if n_ev is None or int(n_ev) < 0:
            # parse from dirname
            name = p.parent.name
            if "_N" in name:
                n_ev = int(name.split("_N")[-1])
            else:
                continue
        rows.append(
            {
                "model": model,
                "n_event": int(n_ev),
                "n_params": int(m.get("n_params") or 0),
                "det_f1": m.get("det_f1"),
                "p_f1": m.get("p_f1"),
                "s_f1": m.get("s_f1"),
                "p_mae_sec": m.get("p_mae_sec"),
                "s_mae_sec": m.get("s_mae_sec"),
                "path": str(p),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = collect_rows(root)
    if not rows:
        print("No test_metrics.json found under", root)
        return

    by_model: dict[str, list] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for m in by_model:
        by_model[m] = sorted(by_model[m], key=lambda x: x["n_event"])

    fits = {}
    efficiency = {}
    for model, rs in by_model.items():
        n = np.array([r["n_event"] for r in rs], dtype=float)
        p_err = 1.0 - np.array([r["p_f1"] for r in rs], dtype=float)
        s_err = 1.0 - np.array([r["s_f1"] for r in rs], dtype=float)
        pick = 0.5 * (
            np.array([r["p_f1"] for r in rs]) + np.array([r["s_f1"] for r in rs])
        )
        fit_p = fit_scaling(n, p_err)
        fit_s = fit_scaling(n, s_err)
        fits[model] = {"p_err": fit_p, "s_err": fit_s, "n_params": rs[-1]["n_params"]}
        # sample efficiency: N to reach F1>=0.90 / 0.95 (err<=0.10/0.05)
        efficiency[model] = {
            "n_params": rs[-1]["n_params"],
            "N_P90": n_to_reach(fit_p, 0.10),
            "N_P95": n_to_reach(fit_p, 0.05),
            "N_S90": n_to_reach(fit_s, 0.10),
            "N_S95": n_to_reach(fit_s, 0.05),
            "alpha_P": fit_p.get("alpha"),
            "alpha_S": fit_s.get("alpha"),
            "r2_P": fit_p.get("r2"),
            "r2_S": fit_s.get("r2"),
            # effective DoF proxy: larger α at similar p → better use of data;
            # p_eff ~ n_params * (α / α_ref) normalized later
            "points": [
                {
                    "n_event": r["n_event"],
                    "p_f1": r["p_f1"],
                    "s_f1": r["s_f1"],
                    "det_f1": r["det_f1"],
                    "p_mae_sec": r["p_mae_sec"],
                    "s_mae_sec": r["s_mae_sec"],
                }
                for r in rs
            ],
        }

    # normalize effective complexity: p_eff = n_params / max(α_P, ε)
    # Higher α = faster error decay = more sample-efficient use of capacity.
    # Report both raw params and α; sample-efficiency ratio vs HNF at F1=0.90.
    board = {
        "protocol": {
            "from_scratch": True,
            "shared_test_subset_seed": 11,
            "tol_sec": 0.5,
            "pick_th": 0.3,
            "fit": "1-F1 = c + a * N^{-alpha}",
        },
        "rows": rows,
        "fits": fits,
        "efficiency": efficiency,
    }
    (root / "SCALING.json").write_text(json.dumps(board, indent=2))

    # Markdown
    lines = [
        "# STEAD scaling law — HNF vs EQTransformer vs PhaseNet",
        "",
        "From-scratch training at matched event budgets; shared eval subset "
        "(seed=11, 8k events + 2k noise); tol=0.5 s, pick_th=0.3.",
        "",
        "## Nominal capacity",
        "",
        "| Model | n_params |",
        "|-------|---------:|",
    ]
    for model, eff in efficiency.items():
        lines.append(f"| {LABEL.get(model, model)} | {eff['n_params']:,} |")

    lines += [
        "",
        "## Sample efficiency (fit \(1-F1=c+a N^{-\\alpha}\))",
        "",
        "| Model | α_P | R²_P | N for P-F1≥0.90 | N for P-F1≥0.95 | α_S | N for S-F1≥0.90 |",
        "|-------|----:|-----:|----------------:|----------------:|----:|----------------:|",
    ]
    for model, eff in efficiency.items():
        def _fmt(x):
            return f"{x:.0f}" if x is not None and math.isfinite(x) else "—"

        lines.append(
            f"| {LABEL.get(model, model)} | "
            f"{eff['alpha_P'] if eff['alpha_P'] is not None else float('nan'):.2f} | "
            f"{eff['r2_P'] if eff['r2_P'] is not None else float('nan'):.2f} | "
            f"{_fmt(eff['N_P90'])} | {_fmt(eff['N_P95'])} | "
            f"{eff['alpha_S'] if eff['alpha_S'] is not None else float('nan'):.2f} | "
            f"{_fmt(eff['N_S90'])} |"
        )

    lines += ["", "## Per-budget metrics", ""]
    for model, rs in by_model.items():
        lines.append(f"### {LABEL.get(model, model)}")
        lines.append("")
        lines.append("| N_event | det F1 | P F1 | S F1 | P MAE | S MAE |")
        lines.append("|--------:|-------:|-----:|-----:|------:|------:|")
        for r in rs:
            lines.append(
                f"| {r['n_event']} | {r['det_f1']:.3f} | {r['p_f1']:.3f} | {r['s_f1']:.3f} | "
                f"{r['p_mae_sec']:.3f} | {r['s_mae_sec']:.3f} |"
            )
        lines.append("")

    lines += [
        "## How to read effective DoF",
        "",
        "- **Nominal DoF** ≈ `n_params` (HNF ~192k, PhaseNet ~268k, EQT ~377k in SeisBench STEAD config).",
        "- **Sample-efficiency exponent α**: larger α → error falls faster with N (better data use).",
        "- **N(F1≥τ)**: data needed to hit a target — primary sample-efficiency score.",
        "- Models with *fewer params but smaller N(F1≥τ)* are more sample-efficient (higher effective capacity per parameter).",
        "",
    ]
    (root / "SCALING.md").write_text("\n".join(lines))

    # Figure
    fig = plt.figure(figsize=(11.0, 7.0), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for model, rs in by_model.items():
        n = [r["n_event"] for r in rs]
        y = [r["p_f1"] for r in rs]
        ax.semilogx(n, y, "o-", color=C.get(model, "k"), label=LABEL.get(model, model), ms=5)
        fit = fits[model]["p_err"]
        if fit.get("ok"):
            ng = np.logspace(np.log10(min(n)), np.log10(max(n)), 50)
            ax.semilogx(ng, 1.0 - power_err(ng, fit["a"], fit["alpha"], fit["c"]), "--", color=C.get(model, "k"), alpha=0.6, lw=1)
    ax.set_xlabel("N_event (train)")
    ax.set_ylabel("P-wave F1")
    ax.set_title("(a) P F1 scaling", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    for model, rs in by_model.items():
        n = [r["n_event"] for r in rs]
        y = [r["s_f1"] for r in rs]
        ax.semilogx(n, y, "o-", color=C.get(model, "k"), label=LABEL.get(model, model), ms=5)
        fit = fits[model]["s_err"]
        if fit.get("ok"):
            ng = np.logspace(np.log10(min(n)), np.log10(max(n)), 50)
            ax.semilogx(ng, 1.0 - power_err(ng, fit["a"], fit["alpha"], fit["c"]), "--", color=C.get(model, "k"), alpha=0.6, lw=1)
    ax.set_xlabel("N_event (train)")
    ax.set_ylabel("S-wave F1")
    ax.set_title("(b) S F1 scaling", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    for model, rs in by_model.items():
        n = [r["n_event"] for r in rs]
        y = [r["p_mae_sec"] for r in rs]
        ax.loglog(n, y, "o-", color=C.get(model, "k"), label=LABEL.get(model, model), ms=5)
    ax.set_xlabel("N_event (train)")
    ax.set_ylabel("P MAE (s)")
    ax.set_title("(c) P timing error", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    txt_lines = ["(d) Capacity vs sample efficiency", ""]
    for model, eff in efficiency.items():
        txt_lines.append(f"{LABEL.get(model, model)}: params={eff['n_params']:,}")
        np90 = eff["N_P90"]
        ap = eff["alpha_P"]
        ap_s = f"{ap:.2f}" if ap is not None else "—"
        n_s = f"{np90:.0f}" if np90 is not None and math.isfinite(np90) else "—"
        txt_lines.append(f"  α_P={ap_s}  N(P≥0.90)={n_s}")
        txt_lines.append("")
    ax.text(0.0, 1.0, "\n".join(txt_lines), va="top", fontsize=9, family="DejaVu Sans")

    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", root / "SCALING.md", f"{out}.png")


if __name__ == "__main__":
    main()
