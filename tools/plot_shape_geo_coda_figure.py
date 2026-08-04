#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figure: morphology classes × geography × coda-Q residual.

Layout (PNF journal style):
  Row 1  a STEAD split 3D pie (w=1) | b geographic map (w=3)
  Row 2  c five morphology-class waveforms (full width)
  Row 3  d class-character cards (w=3) | e coda residual (w=1)

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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  — registers 3d projection
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgb

from hnf.causal_chain import extract_causal_observables, has_valid_chain
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model


# ---- palette (aligned with PNF journal fig1) ----
C_INK = "#0B3D4A"
C_TEAL = "#1B6B93"
C_ACCENT = "#C45C26"
C_MUTED = "#6B7C80"
C_LINE = "#D5DADF"
C_FILL = "#F4F7F8"
BG = "#FFFFFF"

SHAPE_ORDER = (
    "impulsive_fastQ",
    "emergent",
    "multipath",
    "slow_coda",
    "standard",
)

SHAPE_COLORS = {
    "impulsive_fastQ": "#0072B2",
    "emergent": "#E69F00",
    "multipath": "#009E73",
    "slow_coda": "#D55E00",
    "standard": "#7F7F7F",
}

# Curated morphology / ρ / Q captions (visual glyphs carry the numbers).
CLASS_CHAR = {
    "impulsive_fastQ": {
        "morph": "sharp onset · short coda",
        "rho": "tight dual peaks",
        "enrich": "W.US / NE Pac.",
        "Q": "↓Q  faster decay",
        "Q_dir": -1,
    },
    "emergent": {
        "morph": "gradual onset",
        "rho": "broad early rise",
        "enrich": "diffuse",
        "Q": "~Q  mild",
        "Q_dir": 0,
    },
    "multipath": {
        "morph": "multi-arrival coda",
        "rho": "multi-lobe",
        "enrich": "path-diverse",
        "Q": "~Q  baseline",
        "Q_dir": 0,
    },
    "slow_coda": {
        "morph": "long ringing coda",
        "rho": "late sustained",
        "enrich": "C.America / Pac.",
        "Q": "↑Q  slower decay",
        "Q_dir": +1,
    },
    "standard": {
        "morph": "canonical P–S",
        "rho": "clean P/S peaks",
        "enrich": "W.US common",
        "Q": "~Q  reference",
        "Q_dir": 0,
    },
}

# Official STEAD event-trace split sizes (noise excluded; matches dataset loader).
STEAD_SPLITS = {"train": 880875, "val": 46316, "test": 103040}
STEAD_SPLIT_COLORS = {"train": C_TEAL, "val": C_ACCENT, "test": C_MUTED}


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
    p.add_argument("--n-per-class", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--coastline", default="docs/figures/geo/ne_110m_land.geojson")
    p.add_argument("--output", default="docs/figures/seismic_shape_geo_coda.png")
    p.add_argument(
        "--means-cache",
        default="docs/figures/seismic_shape_geo_coda_means.npz",
        help="cache class-mean curves to skip GPU/CPU extract on reruns",
    )
    p.add_argument("--refresh-means", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.linewidth": 0.6,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "text.color": C_INK,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_header(ax, letter: str, title: str, y: float = 1.02, title_dx_pt: float = 14.0) -> None:
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=C_INK,
        va="bottom",
        ha="left",
        zorder=20,
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, y),
        xycoords=ax.transAxes,
        xytext=(title_dx_pt, 0.0),
        textcoords="offset points",
        fontsize=9,
        color=C_INK,
        va="bottom",
        ha="left",
        zorder=20,
        clip_on=False,
    )


def _round_box(ax, xy, w, h, fc, ec, lw=1.0, r=0.012, z=2):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
        mutation_aspect=0.35,
    )
    ax.add_patch(box)
    return box


def _load_coast_polygons(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    polys: list[np.ndarray] = []

    def walk(coords):
        if not coords:
            return
        if isinstance(coords[0][0], (int, float)):
            arr = np.asarray(coords, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                polys.append(arr)
            return
        for c in coords:
            walk(c)

    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        walk(geom.get("coordinates") or [])
    return polys


def _draw_world(ax, coast: list[np.ndarray], *, lon_lim=(-180, 180), lat_lim=(-60, 75)) -> None:
    land = "#E8EEF0"
    edge = "#B4C0C8"
    for ring in coast:
        ax.fill(ring[:, 0], ring[:, 1], facecolor=land, edgecolor=edge, linewidth=0.25, zorder=0)
    ax.set_xlim(*lon_lim)
    ax.set_ylim(*lat_lim)
    # Keep axes box fixed so the map's right edge can align with panel c.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude (°)", color=C_INK)
    ax.set_ylabel("Latitude (°)", color=C_INK)
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_yticks([-40, -20, 0, 20, 40, 60])
    ax.grid(True, which="major", color=C_LINE, linewidth=0.4, linestyle="-", zorder=0.5)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_color("#8A9AA0")


def _pick_exemplars(df: pd.DataFrame, n_per: int, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {}
    for sh in SHAPE_ORDER:
        sub = df[df["shape"] == sh].copy()
        if sub.empty:
            out[sh] = []
            continue
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["trace_name", "dist_km"])
        if "snr_db" in sub.columns:
            sub = sub.sort_values(["snr_db", "dist_km"], ascending=[False, True])
        names = sub["trace_name"].astype(str).tolist()
        if len(names) <= n_per:
            out[sh] = names
        else:
            cand = names[: max(n_per * 8, n_per)]
            idx = rng.choice(len(cand), size=n_per, replace=False)
            out[sh] = [cand[i] for i in sorted(idx)]
    return out


def _to_tau_frame(t: np.ndarray, y: np.ndarray, p_sec: float, s_sec: float, tau_grid: np.ndarray) -> np.ndarray:
    gap = max(float(s_sec - p_sec), 1e-3)
    query = p_sec + tau_grid * gap
    return np.interp(query, t, y, left=y[0], right=y[-1])


def _class_mean_curves(series: dict[str, list[dict]], n: int = 240) -> dict[str, dict]:
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


def _save_means(path: Path, means: dict[str, dict]) -> None:
    payload = {}
    for sh, m in means.items():
        for k, v in m.items():
            payload[f"{sh}__{k}"] = np.asarray(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_means(path: Path) -> dict[str, dict] | None:
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    means: dict[str, dict] = {}
    for sh in SHAPE_ORDER:
        keys = [k for k in data.files if k.startswith(f"{sh}__")]
        if not keys:
            continue
        means[sh] = {k.split("__", 1)[1]: data[k] for k in keys}
        if "n" in means[sh]:
            means[sh]["n"] = int(np.asarray(means[sh]["n"]).reshape(-1)[0])
    return means or None


def _extract_means(args, df: pd.DataFrame) -> dict[str, dict]:
    device = torch.device("cpu" if str(args.device).lower() == "cpu" else args.device)
    exemplars = _pick_exemplars(df, args.n_per_class, args.seed)
    print("[fig] exemplars:", {k: len(v) for k, v in exemplars.items()}, flush=True)
    print(f"[fig] loading checkpoint on {device}…", flush=True)
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    name_index: dict[str, tuple[str, int]] = {}
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
                obs = extract_causal_observables(model, x, t, pick_threshold=0.3, is_event=True)
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
    return _class_mean_curves(series)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def plot_panel_a_pie_3d(fig, host_ax) -> None:
    """Exploded 3D pie; compact legend on top; total count under the pie."""
    host_ax.set_xlim(0, 1)
    host_ax.set_ylim(0, 1)
    host_ax.axis("off")
    host_ax.set_facecolor(BG)

    labels = list(STEAD_SPLITS.keys())
    sizes = np.array([STEAD_SPLITS[k] for k in labels], dtype=float)
    colors = [STEAD_SPLIT_COLORS[k] for k in labels]
    total = float(sizes.sum())
    fracs = sizes / total

    fig.canvas.draw()
    bbox = host_ax.get_position()

    # Compact legend / events strips — leave most of host for a *square* pie.
    leg_h = 0.13 * bbox.height
    bot_h = 0.07 * bbox.height
    # Square side: use nearly full host height and allow spill into panel b.
    side = min(0.98 * bbox.height, 1.65 * bbox.width)
    x0 = bbox.x0 + 0.5 * (bbox.width - side)
    y0 = bbox.y0 + bot_h + 0.5 * max(bbox.height - leg_h - bot_h - side, 0.0)
    # Prefer sitting just above the events strip when square is large.
    if side > bbox.height - leg_h - bot_h:
        y0 = bbox.y0 + bot_h * 0.35
    ax3d = fig.add_axes([x0, y0, side, side], projection="3d", facecolor=BG)
    ax3d.set_zorder(3)

    ax3d.set_axis_off()
    ax3d.set_box_aspect((1.0, 1.0, 0.38))
    ax3d.view_init(elev=24, azim=-55)
    ax3d.set_xlim(-1.25, 1.25)
    ax3d.set_ylim(-1.25, 1.25)
    ax3d.set_zlim(0, 0.5)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BG)
    ax3d.grid(False)

    height = 0.28
    explode = 0.12  # radial offset for exploded wedges
    theta0 = 0.5 * np.pi
    slice_arcs = []
    for frac, col in zip(fracs, colors):
        dtheta = 2.0 * np.pi * float(frac)
        nseg = max(14, int(56 * frac) + 2)
        thetas = np.linspace(theta0, theta0 - dtheta, nseg)
        theta_mid = theta0 - 0.5 * dtheta
        ox = explode * np.cos(theta_mid)
        oy = explode * np.sin(theta_mid)
        slice_arcs.append((thetas, ox, oy))
        rgb = np.array(to_rgb(col))
        top_rgb = np.clip(rgb * 1.06 + 0.04, 0, 1)
        side_rgb = np.clip(rgb * 0.70, 0, 1)

        xt = ox + np.concatenate([[0.0], np.cos(thetas), [0.0]])
        yt = oy + np.concatenate([[0.0], np.sin(thetas), [0.0]])
        zt = np.full_like(xt, height)
        ax3d.add_collection3d(
            Poly3DCollection([list(zip(xt, yt, zt))], facecolors=[top_rgb], edgecolors="white", linewidths=0.35, alpha=1.0)
        )
        for i in range(len(thetas) - 1):
            xa, ya = ox + np.cos(thetas[i]), oy + np.sin(thetas[i])
            xb, yb = ox + np.cos(thetas[i + 1]), oy + np.sin(thetas[i + 1])
            verts = [(xa, ya, 0.0), (xb, yb, 0.0), (xb, yb, height), (xa, ya, height)]
            ax3d.add_collection3d(
                Poly3DCollection([verts], facecolors=[side_rgb], edgecolors=[side_rgb], linewidths=0.04, alpha=1.0)
            )
        for th in (thetas[0], thetas[-1]):
            x, y = ox + np.cos(th), oy + np.sin(th)
            verts = [(ox, oy, 0), (x, y, 0), (x, y, height), (ox, oy, height)]
            ax3d.add_collection3d(
                Poly3DCollection([verts], facecolors=[np.clip(side_rgb * 0.88, 0, 1)], edgecolors="none", alpha=1.0)
            )
        theta0 -= dtheta

    # Horizontal legend ABOVE the pie (within host top band)
    ax_leg = fig.add_axes([bbox.x0, bbox.y0 + bbox.height - leg_h, bbox.width, leg_h], facecolor=BG)
    ax_leg.set_zorder(4)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    # Total count BELOW the pie
    ax_bot = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bot_h], facecolor=BG)
    ax_bot.set_zorder(4)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")
    ax_bot.text(0.5, 0.45, f"{total/1e6:.2f}M events", ha="center", va="center", fontsize=8.0, color=C_INK, fontweight="bold")

    fig.canvas.draw()
    inv = fig.transFigure.inverted()

    ov = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=20)
    ov.set_axis_off()
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.patch.set_alpha(0.0)

    leg_xs = [0.18, 0.50, 0.82]
    for i, lab in enumerate(labels):
        lx = leg_xs[i]
        col = colors[i]
        ax_leg.plot(lx, 0.28, "o", color=col, markersize=7.5, clip_on=False, zorder=3)
        ax_leg.text(lx, 0.72, lab, fontsize=7.6, color=C_INK, va="center", ha="center", fontweight="bold")
        ax_leg.text(
            lx,
            0.50,
            f"{int(sizes[i]):,} ({100.0 * fracs[i]:.1f}%)",
            fontsize=5.6,
            color=C_MUTED,
            va="center",
            ha="center",
        )

        leg_disp = ax_leg.transData.transform((lx, 0.28))
        leg_fig = inv.transform(leg_disp)
        thetas, ox, oy = slice_arcs[i]
        pie_fig = None
        best_d = 1e9
        for th in thetas:
            px, py, pz = ox + 0.96 * np.cos(th), oy + 0.96 * np.sin(th), height
            x2d, y2d, _ = proj3d.proj_transform(px, py, pz, ax3d.get_proj())
            cand = inv.transform(ax3d.transData.transform((x2d, y2d)))
            d = (cand[0] - leg_fig[0]) ** 2 + (cand[1] - leg_fig[1]) ** 2
            if d < best_d:
                best_d = float(d)
                pie_fig = cand
        ov.plot(
            [leg_fig[0], pie_fig[0]],
            [leg_fig[1] - 0.002, pie_fig[1]],
            color=col,
            lw=0.95,
            solid_capstyle="round",
            clip_on=False,
            zorder=21,
        )
        ov.plot(pie_fig[0], pie_fig[1], "o", color=col, markersize=3.0, clip_on=False, zorder=22)


def plot_panel_b_map(ax, df: pd.DataFrame, coast: list[np.ndarray]) -> None:
    _draw_world(ax, coast)

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
            label=sh,
        )

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
        edgecolor=C_LINE,
        framealpha=0.92,
        fontsize=6.5,
    )
    leg.get_frame().set_linewidth(0.5)


def plot_panel_c_morphology(fig, outer_spec, means: dict[str, dict]) -> None:
    """Five stacked axes; class name is each row's y-axis title."""
    gs = GridSpecFromSubplotSpec(
        6,
        1,
        subplot_spec=outer_spec,
        height_ratios=[0.32, 1, 1, 1, 1, 1],
        hspace=0.045,
    )
    ax_h = fig.add_subplot(gs[0])
    ax_h.set_facecolor(BG)
    ax_h.axis("off")
    # Title on the left; legend on the lower-right of the same strip (no overlap).
    _panel_header(ax_h, "c", "Normalised waveforms and ρ(τ) for five morphology classes", y=0.68)
    handles = [
        Line2D([0], [0], color=C_INK, lw=1.2, label="wave envelope"),
        Line2D([0], [0], color=C_INK, lw=1.0, ls=(0, (2.4, 1.5)), label=r"$\rho(\tau)$"),
    ]
    ax_h.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        fontsize=6.3,
        frameon=False,
        bbox_to_anchor=(1.0, 0.02),
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=0.9,
    )

    axes = []
    for i, sh in enumerate(SHAPE_ORDER):
        ax = fig.add_subplot(gs[i + 1], sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.set_facecolor(BG)
        if sh not in means:
            ax.axis("off")
            continue
        m = means[sh]
        c = SHAPE_COLORS[sh]
        tau = m["tau"]
        ax.fill_between(
            tau,
            m["wave_mean"] - m["wave_std"],
            m["wave_mean"] + m["wave_std"],
            color=c,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        ax.plot(tau, m["wave_mean"], color=c, lw=1.2, zorder=2, solid_capstyle="round")
        ax.plot(tau, 0.90 * m["rho_mean"], color=c, lw=1.0, ls=(0, (2.4, 1.5)), alpha=0.95, zorder=3)
        ax.axvline(0.0, color="#8A857A", lw=0.55, ls="-", alpha=0.55, zorder=0)
        ax.axvline(1.0, color="#8A857A", lw=0.55, ls="--", alpha=0.45, zorder=0)
        # Tighten blank between dashed ρ / waves and x-axis to ~1/5 of original (-1.15 → -0.23).
        ax.set_ylim(-0.23, 1.15)
        ax.set_yticks([])
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(C_LINE)
        # class name + n stacked in the upper-left, clear of the curves
        ax.set_ylabel("")
        ax.text(
            0.012,
            0.96,
            f"{sh}\nn={m['n']}",
            transform=ax.transAxes,
            fontsize=7.2,
            color=c,
            fontweight="bold",
            ha="left",
            va="top",
            linespacing=1.15,
            zorder=6,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=BG, edgecolor="none", alpha=0.92),
        )
        # mute the n line by overpainting weight via a second muted pass is hard;
        # keep single block; n uses same color at slightly smaller feel via linespacing.
        if i == 0:
            # P/S inside axes top (below panel title)
            ax.text(0.0, 0.92, "P", transform=ax.get_xaxis_transform(), fontsize=7, color=C_MUTED, ha="center", va="bottom")
            ax.text(1.0, 0.92, "S", transform=ax.get_xaxis_transform(), fontsize=7, color=C_MUTED, ha="center", va="bottom")
        if i < len(SHAPE_ORDER) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Causal time τ  (P = 0, S = 1)", color=C_INK)
    axes[0].set_xlim(-0.25, 2.5)
    return axes


def plot_panel_d_characters(ax, df: pd.DataFrame, means: dict[str, dict]) -> None:
    """Five visual identity cards — no morphology column."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    betas = {sh: float(df.loc[df["shape"] == sh, "coda_path_residual"].median()) for sh in SHAPE_ORDER}
    ns = {sh: int((df["shape"] == sh).sum()) for sh in SHAPE_ORDER}
    beta_max = max(abs(v) for v in betas.values()) + 1e-9

    n_cls = len(SHAPE_ORDER)
    top, bot = 0.97, 0.01
    gap = 0.004
    row_h = (top - bot - (n_cls - 1) * gap) / n_cls

    for i, sh in enumerate(SHAPE_ORDER):
        y1 = top - i * (row_h + gap)
        y0 = y1 - row_h
        yc = 0.5 * (y0 + y1)
        col = SHAPE_COLORS[sh]
        meta = CLASS_CHAR[sh]
        b = betas[sh]
        n = ns[sh]

        _round_box(ax, (0.01, y0), 0.98, row_h, fc="#F7FAFB", ec=col, lw=1.05, r=0.014, z=1)
        ax.add_patch(
            FancyBboxPatch(
                (0.01, y0),
                0.018,
                row_h,
                boxstyle="square,pad=0",
                facecolor=col,
                edgecolor="none",
                zorder=2,
                clip_on=False,
            )
        )
        ax.text(0.040, yc + 0.22 * row_h, sh, fontsize=7.6, color=col, ha="left", va="center", fontweight="bold", zorder=4)
        ax.text(0.040, yc - 0.28 * row_h, f"n = {n}", fontsize=6.2, color=C_MUTED, ha="left", va="center", zorder=4)

        # ρ sparkline: reserved top band for description, tall curve below it
        rho_x0, rho_w = 0.22, 0.30
        ax.text(rho_x0, y1 - 0.012, r"$\rho(\tau)$ shape", fontsize=5.2, color=C_MUTED, ha="left", va="top", zorder=4)
        # blank band for description (above the curve)
        desc_y = y1 - 0.30 * row_h
        ax.text(
            rho_x0 + 0.5 * rho_w,
            desc_y,
            meta["rho"],
            fontsize=5.8,
            color=col,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=6,
        )
        if sh in means:
            tau = np.asarray(means[sh]["tau"], dtype=float)
            rho = np.asarray(means[sh]["rho_mean"], dtype=float)
            rho = rho / (np.max(np.abs(rho)) + 1e-8)
            tt = (tau - tau.min()) / (tau.max() - tau.min() + 1e-9)
            x = rho_x0 + rho_w * tt
            # curve occupies lower ~55% of the card row (taller amplitude)
            curve_bot = y0 + 0.10 * row_h
            curve_top = y1 - 0.42 * row_h
            y = curve_bot + (curve_top - curve_bot) * rho
            ax.fill_between(x, curve_bot, y, color=col, alpha=0.14, lw=0, zorder=3)
            ax.plot(x, y, color=col, lw=1.35, zorder=4)
            ax.plot([rho_x0, rho_x0 + rho_w], [curve_bot, curve_bot], color=C_LINE, lw=0.45, zorder=2)

        # β_res meter
        ax.text(0.55, y1 - 0.018, r"$\beta_{\mathrm{res}}$", fontsize=5.4, color=C_MUTED, ha="left", va="top", zorder=4)
        mx0, mw = 0.55, 0.12
        ax.plot([mx0, mx0 + mw], [yc, yc], color=C_LINE, lw=0.8, zorder=3)
        ax.plot([mx0 + 0.5 * mw] * 2, [yc - 0.028, yc + 0.028], color=C_MUTED, lw=0.7, zorder=3)
        x_mid = mx0 + 0.5 * mw
        x_end = x_mid + 0.5 * mw * (b / beta_max)
        ax.annotate(
            "",
            xy=(x_end, yc),
            xytext=(x_mid, yc),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, mutation_scale=9),
            zorder=5,
        )
        ax.text(mx0 + 0.5 * mw, yc + 0.28 * row_h, f"{b:+.3f}", fontsize=6.4, color=col, ha="center", va="bottom", fontweight="bold", zorder=5)

        # enrichment
        ax.text(0.71, y1 - 0.018, "enrichment", fontsize=5.4, color=C_MUTED, ha="left", va="top", zorder=4)
        ax.plot(0.73, yc + 0.01, "v", color=col, ms=7.5, zorder=5)
        ax.plot(0.73, yc + 0.01, "o", color=col, ms=3.2, zorder=6)
        ax.text(0.75, yc + 0.01, meta["enrich"], fontsize=5.8, color=col, ha="left", va="center", zorder=5)

        # Q
        ax.text(0.90, y1 - 0.018, "Q", fontsize=5.4, color=C_MUTED, ha="center", va="top", zorder=4)
        qd = meta["Q_dir"]
        qx = 0.90
        if qd < 0:
            ax.annotate("", xy=(qx, yc - 0.045), xytext=(qx, yc + 0.045),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7, mutation_scale=10), zorder=5)
        elif qd > 0:
            ax.annotate("", xy=(qx, yc + 0.045), xytext=(qx, yc - 0.045),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7, mutation_scale=10), zorder=5)
        else:
            ax.plot([qx - 0.022, qx + 0.022], [yc, yc], color=col, lw=2.0, solid_capstyle="round", zorder=5)
        ax.text(0.925, yc, meta["Q"].replace("  ", "\n"), fontsize=5.3, color=col, ha="left", va="center", zorder=5)


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


def plot_panel_e_coda(ax, df: pd.DataFrame) -> None:
    data, positions, colors = [], [], []
    for i, sh in enumerate(SHAPE_ORDER):
        vals = df.loc[df["shape"] == sh, "coda_path_residual"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)
        positions.append(i)
        colors.append(SHAPE_COLORS[sh])

    parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showextrema=False, showmedians=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.28)
        body.set_linewidth(0.6)

    for i, (vals, col) in enumerate(zip(data, colors)):
        if vals.size == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(i, q1, q3, color=col, lw=2.2, zorder=3)
        ax.scatter([i], [med], s=18, color="white", edgecolors=col, linewidths=1.0, zorder=4)
        rng = np.random.default_rng(0)
        samp = vals if vals.size <= 80 else rng.choice(vals, 80, replace=False)
        jitter = rng.uniform(-0.12, 0.12, size=samp.size)
        ax.scatter(i + jitter, samp, s=4, color=col, alpha=0.22, linewidths=0, zorder=2, rasterized=True)

    ax.axhline(0.0, color="#9A958A", lw=0.7, ls="--", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(SHAPE_ORDER), rotation=55, ha="right", fontsize=5.8)
    for tick, sh in zip(ax.get_xticklabels(), SHAPE_ORDER):
        tick.set_color(SHAPE_COLORS[sh])
    ax.set_ylabel("Coda path residual", color=C_INK, fontsize=7.5)

    for sh, ha in (("impulsive_fastQ", "left"), ("slow_coda", "right")):
        a = df.loc[df["shape"] == sh, "coda_path_residual"].to_numpy(float)
        b = df.loc[df["shape"] != sh, "coda_path_residual"].to_numpy(float)
        d = _cohen_d(a, b)
        xi = SHAPE_ORDER.index(sh)
        note = "↓Q" if sh == "impulsive_fastQ" else "↑Q"
        ax.annotate(
            f"d={d:+.2f}\n{note}",
            xy=(xi, np.nanmedian(a)),
            xytext=(xi + (0.35 if ha == "left" else -0.35), np.nanmedian(a) + (0.28 if sh == "impulsive_fastQ" else -0.10)),
            fontsize=5.6,
            color=SHAPE_COLORS[sh],
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color=SHAPE_COLORS[sh], lw=0.6),
            zorder=5,
        )

    ax.text(
        0.02,
        0.02,
        "dist.-detrended  ·  Q proxy",
        transform=ax.transAxes,
        fontsize=5.4,
        color=C_MUTED,
        va="bottom",
    )


def main() -> None:
    args = parse_args()
    _style()

    traces_path = Path(args.traces)
    df = pd.read_csv(traces_path)
    if "shape" not in df.columns:
        raise SystemExit("traces CSV missing 'shape' column — run interpretable ceiling first")
    df = df[df["shape"].isin(SHAPE_ORDER)].copy()
    print(f"[fig] loaded {len(df)} labeled traces from {traces_path}", flush=True)

    cache_path = Path(args.means_cache)
    means = None if args.refresh_means else _load_means(cache_path)
    if means is None:
        means = _extract_means(args, df)
        _save_means(cache_path, means)
        print(f"[fig] wrote means cache {cache_path}", flush=True)
    else:
        print(f"[fig] loaded means cache {cache_path}", flush=True)

    coast = _load_coast_polygons(Path(args.coastline))
    print(f"[fig] coastline polygons={len(coast)}", flush=True)

    # Layout: row1 a|b; row2 c waveforms (full width); row3 d characters | e
    fig = plt.figure(figsize=(11.2, 12.8), facecolor=BG)
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.25, 1.50, 1.20],
        hspace=0.20,
        left=0.08,
        right=0.98,
        top=0.955,
        bottom=0.04,
    )

    # Row 1: shared header strip so a/b titles sit at identical height
    gs0 = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs[0],
        height_ratios=[0.18, 1.0],
        width_ratios=[1.35, 2.65],  # wider a → visibly larger square pie
        hspace=0.06,
        wspace=0.10,
    )
    ax_ha = fig.add_subplot(gs0[0, 0])
    ax_hb = fig.add_subplot(gs0[0, 1])
    for axh in (ax_ha, ax_hb):
        axh.set_facecolor(BG)
        axh.axis("off")
    # Titles drawn once here only (do NOT redraw later — that caused ghosting).
    _panel_header(ax_ha, "a", "STEAD dataset split", y=0.20)
    _panel_header(ax_hb, "b", "Geographic distribution — regional enrichment of classes", y=0.20)

    ax_a_host = fig.add_subplot(gs0[1, 0])
    ax_b = fig.add_subplot(gs0[1, 1])

    # Row 2: c morphology waveforms — full width
    axes_c = plot_panel_c_morphology(fig, gs[1], means)

    # Row 3: d characters | e residual — shared header/body height
    gs2 = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs[2],
        height_ratios=[0.16, 1.0],
        width_ratios=[3.0, 1.0],
        hspace=0.06,
        wspace=0.14,
    )
    ax_hd = fig.add_subplot(gs2[0, 0])
    ax_he = fig.add_subplot(gs2[0, 1])
    for axh in (ax_hd, ax_he):
        axh.set_facecolor(BG)
        axh.axis("off")
    _panel_header(ax_hd, "d", "Morphology-class characters", y=0.35)
    _panel_header(ax_he, "e", "Distance-detrended coda residual", y=0.35)
    ax_d = fig.add_subplot(gs2[1, 0])
    ax_e = fig.add_subplot(gs2[1, 1])

    plot_panel_a_pie_3d(fig, ax_a_host)
    plot_panel_b_map(ax_b, df, coast)
    plot_panel_d_characters(ax_d, df, means)
    plot_panel_e_coda(ax_e, df)

    # Keep a/b titles above pie leaders; align b with c; gently compress e.
    fig.canvas.draw()
    for axh in (ax_ha, ax_hb):
        axh.set_zorder(80)
        axh.patch.set_visible(True)
        axh.patch.set_facecolor(BG)
        axh.patch.set_alpha(1.0)
        axh.set_navigate(False)

    pos_c_right = axes_c[0].get_position().x1
    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_b.y0, pos_c_right - pos_b.x0, pos_b.height])
    pos_hb = ax_hb.get_position()
    ax_hb.set_position([pos_b.x0, pos_hb.y0, pos_c_right - pos_b.x0, pos_hb.height])

    pos_d = ax_d.get_position()
    pos_e = ax_e.get_position()
    # Restored toward full d height (was ~0.55–0.78); leave a small gap below.
    e_h = 0.92 * pos_d.height
    ax_e.set_position([pos_e.x0, pos_d.y1 - e_h, pos_e.width, e_h])

    fig.suptitle(
        "Data-driven physics discovery: waveform morphology clusters linked to attenuation structure",
        fontsize=11.5,
        fontweight="bold",
        color=C_INK,
        y=0.985,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=BG, pad_inches=0.04)
    fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi, facecolor=BG, pad_inches=0.04)
    plt.close(fig)
    print(f"[fig] wrote {out}", flush=True)
    print(f"[fig] wrote {out.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
