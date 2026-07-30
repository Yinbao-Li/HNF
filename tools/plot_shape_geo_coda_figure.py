#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figure: morphology classes × geography × coda-Q residual.

Panel (a) normalised waveforms + ρ(t) for five shape classes
Panel (b) geographic enrichment on a world map (lon/lat)
Panel (c) distance-detrended coda residuals (impulsive_fastQ vs slow_coda)

Designed to run on **CPU** so it does not contend with an ongoing GPU train.

Example
-------
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tools/plot_shape_geo_coda_figure.py \\
    --device cpu --traces outputs/interpretable_physics_best/ceiling/traces_labeled.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force CPU before importing torch CUDA contexts elsewhere.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from hnf.causal_chain import extract_causal_observables, has_valid_chain
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model


SHAPE_ORDER = (
    "impulsive_fastQ",
    "emergent",
    "multipath",
    "slow_coda",
    "standard",
)

# Okabe–Ito + muted greys — colourblind-safe, print-friendly
SHAPE_COLORS = {
    "impulsive_fastQ": "#0072B2",
    "emergent": "#E69F00",
    "multipath": "#009E73",
    "slow_coda": "#D55E00",
    "standard": "#7F7F7F",
}

SHAPE_LABELS = {
    "impulsive_fastQ": "impulsive_fastQ",
    "emergent": "emergent",
    "multipath": "multipath",
    "slow_coda": "slow_coda",
    "standard": "standard",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape / geography / coda residual figure")
    p.add_argument(
        "--traces",
        default="outputs/interpretable_physics_best/ceiling/traces_labeled.csv",
    )
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-per-class", type=int, default=8, help="exemplar waveforms per class")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument(
        "--coastline",
        default="docs/figures/geo/ne_110m_land.geojson",
    )
    p.add_argument(
        "--output",
        default="docs/figures/seismic_shape_geo_coda.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_label(ax, letter: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def _load_coast_polygons(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    polys: list[np.ndarray] = []

    def walk(coords, depth: int = 0):
        # GeoJSON Polygon: list of rings; MultiPolygon: list of polygons
        if not coords:
            return
        if isinstance(coords[0][0], (int, float)):
            arr = np.asarray(coords, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                polys.append(arr)
            return
        for c in coords:
            walk(c, depth + 1)

    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        walk(geom.get("coordinates") or [])
    return polys


def _draw_world(ax, coast: list[np.ndarray], *, lon_lim=(-180, 180), lat_lim=(-60, 75)) -> None:
    land = "#ECE9E1"
    edge = "#B7B2A6"
    for ring in coast:
        ax.fill(ring[:, 0], ring[:, 1], facecolor=land, edgecolor=edge, linewidth=0.25, zorder=0)
    ax.set_xlim(*lon_lim)
    ax.set_ylim(*lat_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    # subtle grid
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_yticks([-40, -20, 0, 20, 40, 60])
    ax.grid(True, which="major", color="#D9D4C8", linewidth=0.4, linestyle="-", zorder=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#8A857A")


def _pick_exemplars(df: pd.DataFrame, n_per: int, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {}
    for sh in SHAPE_ORDER:
        sub = df[df["shape"] == sh].copy()
        if sub.empty:
            out[sh] = []
            continue
        # Prefer moderate distance + decent SNR for readable waveforms
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["trace_name", "dist_km"])
        if "snr_db" in sub.columns:
            sub = sub.sort_values(["snr_db", "dist_km"], ascending=[False, True])
        names = sub["trace_name"].astype(str).tolist()
        if len(names) <= n_per:
            out[sh] = names
        else:
            # take top-SNR candidates then sample for diversity
            cand = names[: max(n_per * 8, n_per)]
            idx = rng.choice(len(cand), size=n_per, replace=False)
            out[sh] = [cand[i] for i in sorted(idx)]
    return out


def _to_tau_frame(t: np.ndarray, y: np.ndarray, p_sec: float, s_sec: float, tau_grid: np.ndarray) -> np.ndarray:
    """Resample a physical-time series onto causal tau (P=0, S=1)."""
    gap = max(float(s_sec - p_sec), 1e-3)
    query = p_sec + tau_grid * gap
    return np.interp(query, t, y, left=y[0], right=y[-1])


def _class_mean_curves(series: dict[str, list[dict]], n: int = 240) -> dict[str, dict]:
    """Average exemplars in the causal tau frame (morphology-aligned)."""
    tau_grid = np.linspace(-0.25, 2.5, n)
    out = {}
    for sh, rows in series.items():
        if not rows:
            continue
        W, R = [], []
        for r in rows:
            W.append(_to_tau_frame(r["t"], r["wave"], r["p_sec"], r["s_sec"], tau_grid))
            R.append(_to_tau_frame(r["t"], r["rho"], r["p_sec"], r["s_sec"], tau_grid))
        W = np.stack(W)
        R = np.stack(R)
        Wn = W / (np.max(np.abs(W), axis=1, keepdims=True) + 1e-8)
        Rn = R / (np.max(np.abs(R), axis=1, keepdims=True) + 1e-8)
        out[sh] = {
            "tau": tau_grid,
            "wave_mean": Wn.mean(0),
            "wave_std": Wn.std(0),
            "rho_mean": Rn.mean(0),
            "rho_std": Rn.std(0),
            "n": len(rows),
        }
    return out


def plot_panel_a(ax, means: dict[str, dict]) -> None:
    _panel_label(ax, "a", x=-0.06, y=1.04)
    ax.set_title(
        "Normalised waveforms and ρ(τ) for five morphology classes",
        loc="left",
        pad=6,
    )
    n = len(SHAPE_ORDER)
    gap = 2.45
    yticks, ylabels = [], []
    for i, sh in enumerate(SHAPE_ORDER):
        if sh not in means:
            continue
        m = means[sh]
        y0 = (n - 1 - i) * gap
        c = SHAPE_COLORS[sh]
        tau = m["tau"]
        ax.fill_between(
            tau,
            y0 + m["wave_mean"] - m["wave_std"],
            y0 + m["wave_mean"] + m["wave_std"],
            color=c,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.plot(tau, y0 + m["wave_mean"], color=c, lw=1.25, zorder=2, solid_capstyle="round")
        ax.plot(
            tau,
            y0 + 0.90 * m["rho_mean"],
            color=c,
            lw=1.05,
            ls=(0, (2.4, 1.5)),
            alpha=0.95,
            zorder=3,
        )
        for xv in (0.0, 1.0):
            ax.plot([xv, xv], [y0 - 0.95, y0 + 0.95], color="#B8B2A6", lw=0.55, ls=":", zorder=0)
        yticks.append(y0)
        ylabels.append(SHAPE_LABELS[sh])
        ax.text(
            2.55,
            y0,
            f"n={m['n']}",
            va="center",
            ha="left",
            fontsize=7,
            color="#666666",
            clip_on=False,
        )

    ax.axvline(0.0, color="#8A857A", lw=0.6, ls="-", alpha=0.55, zorder=0)
    ax.axvline(1.0, color="#8A857A", lw=0.6, ls="--", alpha=0.45, zorder=0)
    ax.text(0.02, (n - 1) * gap + 1.15, "P", fontsize=7, color="#666666")
    ax.text(1.02, (n - 1) * gap + 1.15, "S", fontsize=7, color="#666666")
    ax.set_xlim(-0.25, 2.5)
    ax.set_ylim(-1.25, (n - 1) * gap + 1.45)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Causal time τ  (P = 0, S = 1)")
    handles = [
        Line2D([0], [0], color="#333333", lw=1.2, label="wave envelope"),
        Line2D([0], [0], color="#333333", lw=1.0, ls=(0, (2.4, 1.5)), label="ρ(τ)"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.02), ncol=2)
    ax.spines["left"].set_visible(True)


def plot_panel_b(ax, df: pd.DataFrame, coast: list[np.ndarray]) -> None:
    _panel_label(ax, "b", x=-0.08, y=1.06)
    ax.set_title("Geographic distribution — regional enrichment of classes", loc="left", pad=4)
    _draw_world(ax, coast)

    # Draw standard first (background), extremes on top
    order = ["standard", "multipath", "emergent", "slow_coda", "impulsive_fastQ"]
    for sh in order:
        sub = df[df["shape"] == sh].dropna(subset=["src_lon", "src_lat"])
        if sub.empty:
            continue
        size = 10 if sh in ("impulsive_fastQ", "slow_coda") else 7
        alpha = 0.78 if sh in ("impulsive_fastQ", "slow_coda") else 0.38
        z = 4 if sh in ("impulsive_fastQ", "slow_coda") else 2
        ax.scatter(
            sub["src_lon"],
            sub["src_lat"],
            s=size,
            c=SHAPE_COLORS[sh],
            alpha=alpha,
            linewidths=0,
            zorder=z,
            rasterized=True,
            label=SHAPE_LABELS[sh],
        )

    # Enrichment callouts from 10° density peaks on this slice
    callouts = [
        (-125, 35, "impulsive_fastQ\nenriched", SHAPE_COLORS["impulsive_fastQ"], (-95, 52)),
        (-155, 15, "slow_coda\nenriched", SHAPE_COLORS["slow_coda"], (-175, -5)),
    ]
    for lon, lat, text, col, text_xy in callouts:
        ax.annotate(
            text,
            xy=(lon, lat),
            xytext=text_xy,
            fontsize=6.5,
            color=col,
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=2),
            zorder=6,
        )

    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=1,
        markerscale=1.6,
        handletextpad=0.3,
        labelspacing=0.25,
        borderpad=0.3,
        frameon=True,
        fancybox=False,
        edgecolor="#D0CBC0",
        framealpha=0.92,
        fontsize=6.8,
    )
    leg.get_frame().set_linewidth(0.5)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def plot_panel_c(ax, df: pd.DataFrame) -> None:
    _panel_label(ax, "c", x=-0.12, y=1.06)
    ax.set_title("Distance-detrended coda residual (path / Q proxy)", loc="left", pad=4)

    data, positions, colors = [], [], []
    for i, sh in enumerate(SHAPE_ORDER):
        vals = df.loc[df["shape"] == sh, "coda_path_residual"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)
        positions.append(i)
        colors.append(SHAPE_COLORS[sh])

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.28)
        body.set_linewidth(0.6)

    # overlay box-like quartiles + median
    for i, (vals, col) in enumerate(zip(data, colors)):
        if vals.size == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(i, q1, q3, color=col, lw=2.2, zorder=3)
        ax.scatter([i], [med], s=18, color="white", edgecolors=col, linewidths=1.0, zorder=4)
        # faint swarm of a subsample
        rng = np.random.default_rng(0)
        samp = vals if vals.size <= 80 else rng.choice(vals, 80, replace=False)
        jitter = rng.uniform(-0.12, 0.12, size=samp.size)
        ax.scatter(i + jitter, samp, s=4, color=col, alpha=0.22, linewidths=0, zorder=2, rasterized=True)

    ax.axhline(0.0, color="#9A958A", lw=0.7, ls="--", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHAPE_LABELS[s] for s in SHAPE_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Coda path residual")

    # Annotate Cohen's d vs rest for the two extremes
    for sh, ypos, ha in (
        ("impulsive_fastQ", 0.92, "left"),
        ("slow_coda", 0.08, "right"),
    ):
        a = df.loc[df["shape"] == sh, "coda_path_residual"].to_numpy(float)
        b = df.loc[df["shape"] != sh, "coda_path_residual"].to_numpy(float)
        d = _cohen_d(a, b)
        xi = SHAPE_ORDER.index(sh)
        note = "faster decay (↓Q)" if sh == "impulsive_fastQ" else "slower decay (↑Q)"
        ax.annotate(
            f"Cohen's d = {d:+.2f}\n{note}",
            xy=(xi, np.nanmedian(a)),
            xytext=(xi + (0.55 if ha == "left" else -0.55), np.nanmedian(a) + (0.35 if sh == "impulsive_fastQ" else -0.15)),
            fontsize=6.8,
            color=SHAPE_COLORS[sh],
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color=SHAPE_COLORS[sh], lw=0.7),
            zorder=5,
        )

    ax.text(
        0.02,
        0.02,
        "residual after regressing coda slope on distance",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#777777",
        va="bottom",
    )


def main() -> None:
    args = parse_args()
    _style()
    # Always prefer CPU unless the user explicitly requests CUDA (avoids GPU train contention).
    device = torch.device("cpu" if str(args.device).lower() == "cpu" else args.device)

    traces_path = Path(args.traces)
    df = pd.read_csv(traces_path)
    if "shape" not in df.columns:
        raise SystemExit("traces CSV missing 'shape' column — run interpretable ceiling first")
    df = df[df["shape"].isin(SHAPE_ORDER)].copy()
    print(f"[fig] loaded {len(df)} labeled traces from {traces_path}", flush=True)

    exemplars = _pick_exemplars(df, args.n_per_class, args.seed)
    print("[fig] exemplars:", {k: len(v) for k, v in exemplars.items()}, flush=True)

    print(f"[fig] loading checkpoint on {device} (does not use training GPU)…", flush=True)
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    # Ceiling suite used val; also index train so exemplar lookup is robust.
    name_index: dict[str, int] = {}
    ds_by_split: dict[str, STEADPickingDataset] = {}
    for split in ("val", "train"):
        ds = STEADPickingDataset(
            split,
            seq_len=args.seq_len,
            max_event_traces=None,
            max_noise_traces=0,
            seed=args.seed,
        )
        ds_by_split[split] = ds
        for i, ref in enumerate(ds.refs):
            if ref.is_event:
                name_index.setdefault(str(ref.trace_name), (split, i))
    print(f"[fig] STEAD name index size={len(name_index)}", flush=True)

    # Adapt extract to (split, idx) tuples
    series: dict[str, list[dict]] = {k: [] for k in SHAPE_ORDER}
    with torch.no_grad():
        for sh, names in exemplars.items():
            for name in names:
                if name not in name_index:
                    print(f"[fig] missing exemplar {name}", flush=True)
                    continue
                split, i = name_index[name]
                ds = ds_by_split[split]
                batch = ds[i]
                x = batch["x"].unsqueeze(0).to(device)
                t = batch["t"]
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                t = t.to(device)
                obs = extract_causal_observables(
                    model, x, t, pick_threshold=0.3, is_event=True
                )
                if not has_valid_chain(obs):
                    print(f"[fig] invalid chain {name}", flush=True)
                    continue
                wave = np.asarray(obs.wave_env, dtype=float)
                rho = np.asarray(obs.rho, dtype=float)
                wave = wave / (np.max(np.abs(wave)) + 1e-8)
                rho = rho / (np.max(np.abs(rho)) + 1e-8)
                tt = np.linspace(0.0, 60.0, wave.size)
                series[sh].append(
                    {
                        "name": name,
                        "t": tt,
                        "wave": wave,
                        "rho": rho,
                        "p_sec": float(obs.p_sec),
                        "s_sec": float(obs.s_sec),
                    }
                )
    for sh, rows in series.items():
        print(f"  {sh}: extracted {len(rows)}", flush=True)
    means = _class_mean_curves(series)

    coast = _load_coast_polygons(Path(args.coastline))
    print(f"[fig] coastline polygons={len(coast)}", flush=True)

    # --- layout: Nature-style composite ---
    fig = plt.figure(figsize=(7.2, 8.6), dpi=args.dpi)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.15, 1.0],
        hspace=0.28,
        wspace=0.28,
        left=0.10,
        right=0.98,
        top=0.96,
        bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, means)
    plot_panel_b(ax_b, df, coast)
    plot_panel_c(ax_c, df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    # also PDF for journals
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[fig] wrote {out}", flush=True)
    print(f"[fig] wrote {pdf}", flush=True)


if __name__ == "__main__":
    main()
