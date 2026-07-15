#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confirm geo-conditioned rediscovery claims (CPU-only).

Uses existing sample_with_geo.csv — no GPU, no OBS, no STEAD re-index.

Checks:
  1) network confounding (ZQ vs others)
  2) lon-tertile / leave-dominant-cluster sensitivity
  3) stronger controls (distance, depth, SNR, network)
  4) figures for paper-facing strong claims only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_knowledge_mining import (
    _normal_p_from_r,
    bootstrap_ci,
    bootstrap_partial_ci,
    fdr_bh,
    partial_spearman_corr,
    spearman_corr,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confirm geo rediscovery (CPU)")
    p.add_argument(
        "--sample-csv",
        default="outputs/paper_geo_rediscovery/sample_with_geo.csv",
    )
    p.add_argument("--output-dir", default="outputs/paper_geo_confirm")
    p.add_argument("--n-bootstrap", type=int, default=400)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def _series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def pairwise(df: pd.DataFrame, x: str, y: str, n_boot: int, seed: int) -> dict:
    xv, yv = _series(df, x), _series(df, y)
    rho = spearman_corr(xv, yv)
    lo, hi = bootstrap_ci(xv, yv, n_boot=n_boot, seed=seed)
    n = int(np.sum(np.isfinite(xv) & np.isfinite(yv)))
    p = _normal_p_from_r(rho, n)
    return {
        "kind": "pairwise",
        "x": x,
        "y": y,
        "controls": [],
        "stat": float(rho),
        "ci95": [float(lo), float(hi)],
        "n": n,
        "p_approx": float(p),
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }


def partial(
    df: pd.DataFrame, x: str, y: str, controls: list[str], n_boot: int, seed: int
) -> dict:
    xv, yv = _series(df, x), _series(df, y)
    ctr = [_series(df, c) for c in controls]
    rho, n = partial_spearman_corr(xv, yv, ctr)
    lo, hi = bootstrap_partial_ci(xv, yv, ctr, n_boot=n_boot, seed=seed)
    p = _normal_p_from_r(rho, n)
    return {
        "kind": "partial",
        "x": x,
        "y": y,
        "controls": controls,
        "stat": float(rho),
        "ci95": [float(lo), float(hi)],
        "n": int(n),
        "p_approx": float(p),
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }


def mark_supported(recs: list[dict]) -> list[dict]:
    pvals = [1.0 if not np.isfinite(r["p_approx"]) else float(r["p_approx"]) for r in recs]
    qvals = fdr_bh(pvals)
    for r, q in zip(recs, qvals):
        r["fdr_q"] = float(q)
        r["supported"] = bool(r["ci_excludes_zero"] and q <= 0.10)
    return recs


def add_network_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_zq"] = (out["network_code"].astype(str) == "ZQ").astype(float)
    out["is_ta"] = (out["network_code"].astype(str) == "TA").astype(float)
    # dominant ZQ lobe vs rest (approx from sample medians)
    out["in_zq_box"] = (
        (out["source_lat"] < -20.0) & (out["source_lon"].between(20.0, 40.0))
    ).astype(float)
    return out


def run_core_battery(df: pd.DataFrame, n_boot: int, seed: int, scope: str) -> list[dict]:
    """Battery focused on strong / priority claims."""
    specs = []
    # absolute geo (pairwise + distance/depth partial)
    for x in ("source_lat", "source_lon", "receiver_lat", "receiver_lon", "back_azimuth"):
        for y in ("pick_err_p", "pick_err_s", "init_tt", "rho_mean", "noise_ratio"):
            if (x, y) in {
                ("source_lat", "pick_err_p"),
                ("source_lon", "pick_err_p"),
                ("source_lon", "pick_err_s"),
                ("source_lat", "init_tt"),
                ("source_lon", "init_tt"),
                ("source_lat", "rho_mean"),
                ("source_lon", "rho_mean"),
                ("source_lat", "noise_ratio"),
                ("source_lon", "noise_ratio"),
                ("receiver_lat", "pick_err_p"),
                ("receiver_lon", "pick_err_p"),
                ("back_azimuth", "pick_err_p"),
            }:
                specs.append(("pair", x, y, []))
                specs.append(("part", x, y, ["distance_km", "source_depth_km"]))

    # priority laws with geography / network controls
    priority = [
        ("noise_ratio", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("noise_ratio", "pick_err_p", ["distance_km", "source_depth_km", "source_lat", "source_lon"]),
        ("noise_ratio", "pick_err_p", ["distance_km", "source_depth_km", "is_zq"]),
        ("noise_ratio", "pick_err_p", ["distance_km", "source_depth_km", "snr_db"]),
        ("rho_p_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        (
            "rho_p_lag",
            "init_tt",
            ["distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon"],
        ),
        ("rho_p_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_p", "is_zq"]),
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km"]),
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km", "source_lat", "source_lon"]),
        ("rho_mean", "vp_mean", ["distance_km", "source_depth_km", "is_zq"]),
        # network as predictor
        ("is_zq", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("is_zq", "noise_ratio", ["distance_km", "source_depth_km"]),
        ("is_zq", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        # absolute geo after network control
        ("source_lat", "pick_err_p", ["distance_km", "source_depth_km", "is_zq"]),
        ("source_lon", "pick_err_p", ["distance_km", "source_depth_km", "is_zq"]),
        ("source_lat", "init_tt", ["distance_km", "source_depth_km", "pick_err_p", "is_zq"]),
        ("source_lon", "init_tt", ["distance_km", "source_depth_km", "pick_err_p", "is_zq"]),
        ("source_lat", "noise_ratio", ["distance_km", "source_depth_km", "is_zq"]),
        ("source_lat", "rho_mean", ["distance_km", "source_depth_km", "is_zq"]),
    ]

    recs = []
    i = 0
    for kind, x, y, ctrl in specs:
        if kind == "pair":
            rec = pairwise(df, x, y, n_boot, seed + i)
        else:
            rec = partial(df, x, y, ctrl, n_boot, seed + i)
        rec["scope"] = scope
        recs.append(rec)
        i += 1
    for x, y, ctrl in priority:
        rec = partial(df, x, y, ctrl, n_boot, seed + i)
        rec["scope"] = scope
        recs.append(rec)
        i += 1
    return mark_supported(recs)


def claim_table(recs: list[dict], keys: list[tuple]) -> list[dict]:
    out = []
    for kind, x, y, ctrl in keys:
        ctrl_t = tuple(ctrl)
        hits = [
            r
            for r in recs
            if r["kind"] == kind
            and r["x"] == x
            and r["y"] == y
            and tuple(r["controls"]) == ctrl_t
        ]
        if hits:
            out.append(hits[0])
    return out


def plot_strong_claims(df: pd.DataFrame, confirm: dict, out_dir: Path, docs_dir: Path) -> list[str]:
    """Three paper figures for strong claims only."""
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Fig A: map colored by pick_err_p + network
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, ycol, title in zip(
        axes,
        ("pick_err_p", "noise_ratio"),
        ("P pick error vs source location", "noise_ratio vs source location"),
    ):
        yv = _series(df, ycol)
        # clip for visibility
        lo, hi = np.nanpercentile(yv, [5, 95])
        sc = ax.scatter(
            df["source_lon"],
            df["source_lat"],
            c=np.clip(yv, lo, hi),
            s=22,
            cmap="viridis",
            alpha=0.85,
            edgecolors="none",
        )
        # mark non-ZQ
        m = df["network_code"].astype(str) != "ZQ"
        ax.scatter(
            df.loc[m, "source_lon"],
            df.loc[m, "source_lat"],
            facecolors="none",
            edgecolors="crimson",
            s=55,
            linewidths=0.9,
            label="non-ZQ",
        )
        ax.set_xlabel("source longitude")
        ax.set_ylabel("source latitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Absolute geography carries QC signal (n={len(df)}; ZQ={int((df.network_code=='ZQ').sum())})",
        fontsize=11,
    )
    fig.tight_layout()
    p = out_dir / "geo_qc_spatial_map.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    shutil.copy2(p, docs_dir / p.name)
    written.append(str(docs_dir / p.name))

    # Fig B: priority laws survive geography / network controls
    keys = [
        ("partial", "noise_ratio", "pick_err_p", ["distance_km", "source_depth_km"]),
        (
            "partial",
            "noise_ratio",
            "pick_err_p",
            ["distance_km", "source_depth_km", "source_lat", "source_lon"],
        ),
        ("partial", "noise_ratio", "pick_err_p", ["distance_km", "source_depth_km", "is_zq"]),
        ("partial", "rho_p_lag", "init_tt", ["distance_km", "source_depth_km", "pick_err_p"]),
        (
            "partial",
            "rho_p_lag",
            "init_tt",
            ["distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon"],
        ),
        (
            "partial",
            "rho_p_lag",
            "init_tt",
            ["distance_km", "source_depth_km", "pick_err_p", "is_zq"],
        ),
        ("partial", "rho_mean", "vp_mean", ["distance_km", "source_depth_km"]),
        (
            "partial",
            "rho_mean",
            "vp_mean",
            ["distance_km", "source_depth_km", "source_lat", "source_lon"],
        ),
        ("partial", "rho_mean", "vp_mean", ["distance_km", "source_depth_km", "is_zq"]),
    ]
    rows = claim_table(confirm["global"], keys)
    labels = [
        "noise→err\n+dist/dep",
        "noise→err\n+lat/lon",
        "noise→err\n+is_ZQ",
        "ρ_lag→TT\n+geom/err",
        "ρ_lag→TT\n+lat/lon",
        "ρ_lag→TT\n+is_ZQ",
        "ρ→Vp\n+dist/dep",
        "ρ→Vp\n+lat/lon",
        "ρ→Vp\n+is_ZQ",
    ]
    stats = [r["stat"] for r in rows]
    los = [r["ci95"][0] for r in rows]
    his = [r["ci95"][1] for r in rows]
    supported = [r["supported"] for r in rows]
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if s else "#bdbdbd" for s in supported]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x, stats, color=colors, alpha=0.9, edgecolor="none")
    ax.errorbar(
        x,
        stats,
        yerr=[np.array(stats) - np.array(los), np.array(his) - np.array(stats)],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=1.0,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("partial Spearman ρ (95% bootstrap CI)")
    ax.set_title("Priority laws remain after absolute-geo / network controls")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "geo_priority_controls.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    shutil.copy2(p, docs_dir / p.name)
    written.append(str(docs_dir / p.name))

    # Fig C: absolute-geo predictors — raw vs network-controlled
    keys2 = [
        ("pairwise", "source_lat", "pick_err_p", []),
        ("partial", "source_lat", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("partial", "source_lat", "pick_err_p", ["distance_km", "source_depth_km", "is_zq"]),
        ("pairwise", "source_lat", "init_tt", []),
        ("partial", "source_lat", "init_tt", ["distance_km", "source_depth_km", "pick_err_p", "is_zq"]),
        ("pairwise", "source_lon", "pick_err_s", []),
        ("partial", "is_zq", "pick_err_p", ["distance_km", "source_depth_km"]),
        ("partial", "is_zq", "noise_ratio", ["distance_km", "source_depth_km"]),
    ]
    rows2 = claim_table(confirm["global"], keys2)
    labels2 = [
        "lat→Perr\npair",
        "lat→Perr\n+dist/dep",
        "lat→Perr\n+is_ZQ",
        "lat→TT\npair",
        "lat→TT\n+ZQ",
        "lon→Serr\npair",
        "ZQ→Perr\npartial",
        "ZQ→noise\npartial",
    ]
    stats2 = [r["stat"] for r in rows2]
    los2 = [r["ci95"][0] for r in rows2]
    his2 = [r["ci95"][1] for r in rows2]
    supported2 = [r["supported"] for r in rows2]
    x2 = np.arange(len(rows2))
    colors2 = ["#e76f51" if s else "#bdbdbd" for s in supported2]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(x2, stats2, color=colors2, alpha=0.9, edgecolor="none")
    ax.errorbar(
        x2,
        stats2,
        yerr=[np.array(stats2) - np.array(los2), np.array(his2) - np.array(stats2)],
        fmt="none",
        ecolor="#333333",
        capsize=3,
        lw=1.0,
    )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2, fontsize=8)
    ax.set_ylabel("Spearman / partial ρ (95% CI)")
    ax.set_title("Absolute-geo edges: many collapse after network (ZQ) control")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "geo_absolute_vs_network.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    shutil.copy2(p, docs_dir / p.name)
    written.append(str(docs_dir / p.name))

    # Fig D: sensitivity — global / non-ZQ / leave-C3 / lon tertiles
    sens = confirm["sensitivity"]
    names = list(sens.keys())
    # pick three claims across scopes
    claim_ids = [
        ("noise_ratio", "pick_err_p", ("distance_km", "source_depth_km", "source_lat", "source_lon")),
        ("rho_p_lag", "init_tt", ("distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon")),
        ("source_lat", "pick_err_p", ("distance_km", "source_depth_km")),
    ]
    claim_labels = ["noise→err | +lat/lon", "ρ_lag→TT | +lat/lon", "lat→Perr | +dist/dep"]
    mat = np.full((len(claim_ids), len(names)), np.nan)
    support = np.zeros_like(mat, dtype=bool)
    for j, name in enumerate(names):
        for i, (x, y, ctrl) in enumerate(claim_ids):
            for r in sens[name]:
                if r["x"] == x and r["y"] == y and tuple(r["controls"]) == ctrl and r["kind"] == "partial":
                    mat[i, j] = r["stat"]
                    support[i, j] = r["supported"]
                    break
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(claim_labels)))
    ax.set_yticklabels(claim_labels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                mark = "*" if support[i, j] else ""
                ax.text(j, i, f"{mat[i, j]:.2f}{mark}", ha="center", va="center", fontsize=8)
    ax.set_title("Sensitivity of key partials (* = CI∌0 & FDR q≤0.10)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    p = out_dir / "geo_sensitivity_heatmap.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    shutil.copy2(p, docs_dir / p.name)
    written.append(str(docs_dir / p.name))

    return written


def write_md(report: dict, path: Path) -> None:
    g = { (r["kind"], r["x"], r["y"], tuple(r["controls"])): r for r in report["global"] }
    lines = [
        "# Geo rediscovery confirmation",
        "",
        f"- n={report['n']}",
        f"- networks: {report['network_counts']}",
        f"- geo_cluster sizes: {report['geo_cluster_sizes']}",
        "",
        "## Verdicts",
        "",
    ]
    for v in report["verdicts"]:
        lines.append(f"- **{v['id']}**: {v['label']} — {v['note']}")
    lines += ["", "## Key partials (global)", ""]
    for key in [
        ("partial", "noise_ratio", "pick_err_p", ("distance_km", "source_depth_km", "source_lat", "source_lon")),
        ("partial", "rho_p_lag", "init_tt", ("distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon")),
        ("partial", "rho_mean", "vp_mean", ("distance_km", "source_depth_km", "source_lat", "source_lon")),
        ("partial", "source_lat", "pick_err_p", ("distance_km", "source_depth_km")),
        ("partial", "source_lat", "pick_err_p", ("distance_km", "source_depth_km", "is_zq")),
        ("partial", "is_zq", "pick_err_p", ("distance_km", "source_depth_km")),
        ("partial", "is_zq", "noise_ratio", ("distance_km", "source_depth_km")),
    ]:
        r = g.get(key)
        if not r:
            continue
        lines.append(
            f"- `{r['x']}→{r['y']}` ctrl={list(r['controls'])}: "
            f"ρ={r['stat']:.3f} CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
            f"supported={r['supported']}"
        )
    lines += ["", "## Figures", ""]
    for f in report["figures"]:
        lines.append(f"- `{f}`")
    path.write_text("\n".join(lines) + "\n")


def build_verdicts(global_recs: list[dict], sens: dict) -> list[dict]:
    g = {(r["kind"], r["x"], r["y"], tuple(r["controls"])): r for r in global_recs}

    def get(*key):
        return g.get(key)

    verdicts = []

    r1 = get("partial", "noise_ratio", "pick_err_p", ("distance_km", "source_depth_km", "source_lat", "source_lon"))
    r1b = get("partial", "noise_ratio", "pick_err_p", ("distance_km", "source_depth_km", "is_zq"))
    if r1 and r1["supported"] and r1b and r1b["supported"]:
        verdicts.append(
            {
                "id": "noise_ratio→pick_err_p",
                "label": "CONFIRMED (strong)",
                "note": (
                    f"survives lat/lon (ρ={r1['stat']:.3f}) and is_ZQ (ρ={r1b['stat']:.3f}); "
                    "not a geography artifact"
                ),
            }
        )
    else:
        verdicts.append(
            {
                "id": "noise_ratio→pick_err_p",
                "label": "WEAKENED",
                "note": "failed under geography or network control",
            }
        )

    r2 = get(
        "partial",
        "rho_p_lag",
        "init_tt",
        ("distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon"),
    )
    r2b = get("partial", "rho_p_lag", "init_tt", ("distance_km", "source_depth_km", "pick_err_p", "is_zq"))
    if r2 and r2["supported"] and r2b and r2b["supported"]:
        verdicts.append(
            {
                "id": "rho_p_lag→init_tt",
                "label": "CONFIRMED (strong)",
                "note": (
                    f"survives lat/lon (ρ={r2['stat']:.3f}) and is_ZQ (ρ={r2b['stat']:.3f})"
                ),
            }
        )
    else:
        verdicts.append(
            {
                "id": "rho_p_lag→init_tt",
                "label": "WEAKENED",
                "note": "failed under geography or network control",
            }
        )

    r3 = get("partial", "rho_mean", "vp_mean", ("distance_km", "source_depth_km", "source_lat", "source_lon"))
    r3b = get("partial", "rho_mean", "vp_mean", ("distance_km", "source_depth_km", "is_zq"))
    if r3 and r3["supported"]:
        note = f"survives lat/lon (ρ={r3['stat']:.3f})"
        if r3b and r3b["supported"]:
            note += f"; also survives is_ZQ (ρ={r3b['stat']:.3f})"
        else:
            note += "; network control weakens/removes support — keep as secondary"
        verdicts.append({"id": "rho_mean→vp_mean", "label": "CONFIRMED (moderate)", "note": note})
    else:
        verdicts.append(
            {
                "id": "rho_mean→vp_mean",
                "label": "NOT CONFIRMED under geo",
                "note": "does not survive absolute-geo controls in this battery",
            }
        )

    r4 = get("partial", "source_lat", "pick_err_p", ("distance_km", "source_depth_km"))
    r4b = get("partial", "source_lat", "pick_err_p", ("distance_km", "source_depth_km", "is_zq"))
    r_net = get("partial", "is_zq", "pick_err_p", ("distance_km", "source_depth_km"))
    if r4 and r4["supported"] and r4b and not r4b["supported"]:
        verdicts.append(
            {
                "id": "source_lat→pick_err_p",
                "label": "REINTERPRETED",
                "note": (
                    f"raw/geo-partial supported (ρ={r4['stat']:.3f}) but collapses after is_ZQ; "
                    f"network effect is_ZQ→pick_err_p ρ={r_net['stat']:.3f} "
                    f"supported={r_net['supported'] if r_net else None}. "
                    "Treat as regional/network geography, not latitude physics."
                ),
            }
        )
    elif r4 and r4["supported"] and r4b and r4b["supported"]:
        verdicts.append(
            {
                "id": "source_lat→pick_err_p",
                "label": "CONFIRMED within-network",
                "note": "survives is_ZQ control — residual absolute-geo signal",
            }
        )
    else:
        verdicts.append(
            {
                "id": "source_lat→pick_err_p",
                "label": "UNSUPPORTED",
                "note": "not stable in confirmation battery",
            }
        )

    # sensitivity: priority should hold on non-ZQ and leave-C3 if possible
    def sens_ok(name: str, x: str, y: str, ctrl: tuple) -> bool | None:
        for r in sens.get(name, []):
            if r["x"] == x and r["y"] == y and tuple(r["controls"]) == ctrl and r["kind"] == "partial":
                return bool(r["supported"])
        return None

    ctrl_noise = ("distance_km", "source_depth_km", "source_lat", "source_lon")
    ctrl_lag = ("distance_km", "source_depth_km", "pick_err_p", "source_lat", "source_lon")
    leave = sens_ok("leave_C3", "noise_ratio", "pick_err_p", ctrl_noise)
    nonzq = sens_ok("non_ZQ", "noise_ratio", "pick_err_p", ctrl_noise)
    leave2 = sens_ok("leave_C3", "rho_p_lag", "init_tt", ctrl_lag)
    nonzq2 = sens_ok("non_ZQ", "rho_p_lag", "init_tt", ctrl_lag)
    verdicts.append(
        {
            "id": "sensitivity",
            "label": "REPORTED",
            "note": (
                f"noise→err leave_C3={leave}, non_ZQ={nonzq}; "
                f"ρ_lag→TT leave_C3={leave2}, non_ZQ={nonzq2}. "
                "Small non-ZQ / leave-C3 n limits power — absence of support ≠ rejection."
            ),
        }
    )
    return verdicts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs/figures")

    df = pd.read_csv(args.sample_csv)
    df = add_network_flags(df)
    print(
        f"[confirm] n={len(df)} networks={df.network_code.value_counts().to_dict()} "
        f"clusters={df.geo_cluster.value_counts().to_dict()}",
        flush=True,
    )

    global_recs = run_core_battery(df, args.n_bootstrap, args.seed, "global")
    print(f"[confirm] global supported={sum(r['supported'] for r in global_recs)}/{len(global_recs)}", flush=True)

    sensitivity: dict[str, list[dict]] = {"global": global_recs}
    # leave dominant geo cluster
    if "geo_cluster" in df.columns:
        dominant = int(df["geo_cluster"].value_counts().idxmax())
        sub = df[df["geo_cluster"] != dominant]
        if len(sub) >= 40:
            sensitivity["leave_C3"] = run_core_battery(
                sub, args.n_bootstrap, args.seed + 50, f"leave_C{dominant}"
            )
            print(f"[confirm] leave_C{dominant} n={len(sub)}", flush=True)
    # non-ZQ only
    sub = df[df["network_code"].astype(str) != "ZQ"]
    if len(sub) >= 40:
        sensitivity["non_ZQ"] = run_core_battery(sub, args.n_bootstrap, args.seed + 90, "non_ZQ")
        print(f"[confirm] non_ZQ n={len(sub)}", flush=True)
    # ZQ-only (within-network absolute geo)
    sub = df[df["network_code"].astype(str) == "ZQ"]
    if len(sub) >= 40:
        sensitivity["ZQ_only"] = run_core_battery(sub, args.n_bootstrap, args.seed + 130, "ZQ_only")
        print(f"[confirm] ZQ_only n={len(sub)}", flush=True)
    # lon tertiles
    for t in sorted(df["lon_tertile"].dropna().unique()):
        sub = df[df["lon_tertile"] == t]
        if len(sub) >= 40:
            name = f"lon_T{int(t)}"
            sensitivity[name] = run_core_battery(sub, args.n_bootstrap, args.seed + 200 + int(t), name)
            print(f"[confirm] {name} n={len(sub)}", flush=True)

    verdicts = build_verdicts(global_recs, sensitivity)
    report = {
        "n": int(len(df)),
        "network_counts": {str(k): int(v) for k, v in df.network_code.value_counts().items()},
        "geo_cluster_sizes": {str(k): int(v) for k, v in df.geo_cluster.value_counts().items()},
        "global": global_recs,
        "sensitivity": {k: v for k, v in sensitivity.items()},
        "verdicts": verdicts,
        "figures": [],
    }
    figs = plot_strong_claims(df, report, out_dir, docs_dir)
    report["figures"] = figs

    (out_dir / "geo_confirm_report.json").write_text(json.dumps(report, indent=2))
    write_md(report, out_dir / "geo_confirm_report.md")
    print(json.dumps(verdicts, indent=2), flush=True)
    print(f"[confirm] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
