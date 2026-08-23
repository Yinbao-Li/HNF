#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified-propagation Fig. 2 — seismic facies + SoCal path residuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEAD_DIR

C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
BG = "#FFFFFF"
C_ABS = "#B91C1C"
C_RING = "#1D4ED8"

SHAPE_ORDER = ["impulsive_fastQ", "emergent", "multipath", "slow_coda", "standard"]
SHAPE_COLORS = {
    "impulsive_fastQ": "#0072B2",
    "emergent": "#E69F00",
    "multipath": "#009E73",
    "slow_coda": "#D55E00",
    "standard": "#6B7280",
}
SHAPE_LABEL = {
    "impulsive_fastQ": "impulsive fast-Q",
    "emergent": "emergent",
    "multipath": "multipath",
    "slow_coda": "slow coda",
    "standard": "standard",
}
SHAPE_TAG = {
    "impulsive_fastQ": "sharp + steep coda",
    "emergent": "slow onset",
    "multipath": "multi-lobe ρ",
    "slow_coda": "late energy",
    "standard": "baseline",
}
CELLS = [
    {"rank": 2, "lon": -116.375, "lat": 33.375, "short": "Salton", "color": C_ABS, "note": "faster decay"},
    {"rank": 5, "lon": -116.375, "lat": 33.875, "short": "E. Transverse R.", "color": C_RING, "note": "longer ringing"},
]
GRID = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/figures/unified")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--socal-dir", default="outputs/structure_residual_socal")
    p.add_argument("--means", default="docs/figures/seismic_shape_geo_coda_means.npz")
    p.add_argument("--panel", default="outputs/shape_labels_expanded/socal/panel_selection.csv")
    p.add_argument("--stead-dir", default=str(STEAD_DIR))
    p.add_argument("--n-stations", type=int, default=4)
    p.add_argument("--max-per-arm", type=int, default=24)
    p.add_argument("--rebuild-stacks", action="store_true")
    return p.parse_args()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": BG,
        "figure.facecolor": BG,
    })


def clean(ax) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, colors=C_INK)


def titles(fig, items, *, y_top: float, y_bot: float) -> None:
    for i, (ax_ref, letter, title) in enumerate(items):
        axes = ax_ref if isinstance(ax_ref, (list, tuple)) else [ax_ref]
        x0 = min(a.get_position().x0 for a in axes)
        y = y_top if i < 2 else y_bot
        fig.text(x0, y, letter, fontsize=11, fontweight="bold", color=C_INK, va="bottom", ha="left")
        fig.text(x0 + 0.018, y, title, fontsize=8.5, color=C_INK, va="bottom", ha="left")


def load_lines(path: Path) -> list:
    if not path.is_file():
        return []
    geo = json.loads(path.read_text(encoding="utf-8"))
    lines = []

    def add(coords):
        arr = np.asarray(coords, float)
        if arr.ndim == 2 and len(arr) >= 2:
            lines.append(arr)

    for feat in geo.get("features", []):
        g = feat.get("geometry") or {}
        t, c = g.get("type"), g.get("coordinates")
        if t == "LineString":
            add(c)
        elif t == "MultiLineString":
            for part in c:
                add(part)
        elif t == "Polygon":
            add(c[0])
        elif t == "MultiPolygon":
            for poly in c:
                add(poly[0])
    return lines


def load_means(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    means = {}
    for sh in SHAPE_ORDER:
        keys = [k for k in data.files if k.startswith(f"{sh}__")]
        if not keys:
            continue
        means[sh] = {k.split("__", 1)[1]: data[k] for k in keys}
        if "n" in means[sh]:
            means[sh]["n"] = int(np.asarray(means[sh]["n"]).reshape(-1)[0])
    if len(means) < len(SHAPE_ORDER):
        raise FileNotFoundError(f"incomplete facies means in {path}: got {list(means)}")
    return means


def cell_key(lon: float, lat: float):
    gx = np.floor(lon / GRID) * GRID + 0.5 * GRID
    gy = np.floor(lat / GRID) * GRID + 0.5 * GRID
    return round(float(gx), 6), round(float(gy), 6)


def envelope(wave: np.ndarray) -> np.ndarray:
    e = np.sqrt(np.mean(np.square(wave.astype(np.float64)), axis=1) + 1e-12)
    k = np.ones(21) / 21.0
    return np.convolve(e, k, mode="same")


def s_aligned_env(wave, s_sample, t_grid):
    if not np.isfinite(s_sample):
        return None
    n = wave.shape[0]
    t_abs = (np.arange(n, dtype=float) - float(s_sample)) / 100.0
    env = envelope(wave)
    if t_abs[0] > -1.0 or t_abs[-1] < 20.0:
        return None
    y = np.interp(t_grid, t_abs, env, left=np.nan, right=np.nan)
    if not np.isfinite(y).any():
        return None
    m = (t_grid >= 0.0) & (t_grid <= 8.0) & np.isfinite(y)
    if m.sum() < 10:
        return None
    peak = float(np.nanmax(y[m]))
    if peak <= 1e-9:
        return None
    return y / peak


def stack_stats(mat: np.ndarray) -> dict:
    return {
        "mean": np.nanmean(mat, axis=0),
        "p25": np.nanpercentile(mat, 25, axis=0),
        "p75": np.nanpercentile(mat, 75, axis=0),
        "n": np.array(mat.shape[0], dtype=np.int32),
    }


def build_or_load_stacks(cache, *, traces, panel_csv, bysite, stead_dir, n_stations, max_per_arm, rebuild):
    if cache.is_file() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        out = {"t": z["t"]}
        for cell in CELLS:
            key = f"rank{cell['rank']}"
            out[key] = {
                "in_mean": z[f"{key}__in_mean"],
                "in_p25": z[f"{key}__in_p25"],
                "in_p75": z[f"{key}__in_p75"],
                "out_mean": z[f"{key}__out_mean"],
                "out_p25": z[f"{key}__out_p25"],
                "out_p75": z[f"{key}__out_p75"],
                "n_in": int(z[f"{key}__n_in"]),
                "n_out": int(z[f"{key}__n_out"]),
                "n_sites": int(z[f"{key}__n_sites"]),
            }
        return out

    df = pd.read_csv(traces)
    panel = pd.read_csv(panel_csv, usecols=["trace_name", "chunk", "s_arrival_sample", "site"])
    site_tbl = pd.read_csv(bysite)
    meta = panel.drop_duplicates("trace_name").set_index("trace_name")
    df["gx"], df["gy"] = zip(*[cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])

    t_grid = np.linspace(-2.0, 28.0, 601)
    handles = {}

    def open_h5(chunk: int):
        if chunk not in handles:
            handles[chunk] = h5py.File(stead_dir / f"chunk{chunk}_eofextract" / f"chunk{chunk}.hdf5", "r")
        return handles[chunk]

    payload = {"t": t_grid}
    result = {"t": t_grid}

    for cell in CELLS:
        sites = (
            site_tbl[site_tbl["rank"] == cell["rank"]]
            .assign(absd=lambda d: np.abs(d["delta"]))
            .sort_values("absd", ascending=False)
        )
        sites = sites[sites["n_in"] >= 3].head(n_stations)
        in_m = np.isclose(df["gx"], cell["lon"]) & np.isclose(df["gy"], cell["lat"])
        buf = 0.5 * GRID
        near = (np.abs(df["gx"] - cell["lon"]) <= buf + 1e-9) & (np.abs(df["gy"] - cell["lat"]) <= buf + 1e-9)

        env_in, env_out = [], []
        for site in sites["site"].astype(str):
            sub_in = df[in_m & (df["site"] == site)]
            sub_out = df[(~near) & (df["site"] == site)]
            for _, row in sub_in.head(max_per_arm).iterrows():
                name = str(row["trace_name"])
                if name not in meta.index:
                    continue
                rec = meta.loc[name]
                try:
                    w = open_h5(int(rec["chunk"]))["data"][name][()]
                except Exception:
                    continue
                y = s_aligned_env(w, float(rec["s_arrival_sample"]), t_grid)
                if y is not None:
                    env_in.append(y)
            for _, row in sub_out.head(max_per_arm).iterrows():
                name = str(row["trace_name"])
                if name not in meta.index:
                    continue
                rec = meta.loc[name]
                try:
                    w = open_h5(int(rec["chunk"]))["data"][name][()]
                except Exception:
                    continue
                y = s_aligned_env(w, float(rec["s_arrival_sample"]), t_grid)
                if y is not None:
                    env_out.append(y)

        if len(env_in) < 5 or len(env_out) < 5:
            raise RuntimeError(f"insufficient stacks for rank {cell['rank']}: in={len(env_in)} out={len(env_out)}")

        inn = stack_stats(np.asarray(env_in))
        out = stack_stats(np.asarray(env_out))
        key = f"rank{cell['rank']}"
        result[key] = {
            "in_mean": inn["mean"], "in_p25": inn["p25"], "in_p75": inn["p75"],
            "out_mean": out["mean"], "out_p25": out["p25"], "out_p75": out["p75"],
            "n_in": int(inn["n"]), "n_out": int(out["n"]), "n_sites": int(len(sites)),
        }
        for arm, st in (("in", inn), ("out", out)):
            payload[f"{key}__{arm}_mean"] = st["mean"]
            payload[f"{key}__{arm}_p25"] = st["p25"]
            payload[f"{key}__{arm}_p75"] = st["p75"]
        payload[f"{key}__n_in"] = np.asarray(inn["n"])
        payload[f"{key}__n_out"] = np.asarray(out["n"])
        payload[f"{key}__n_sites"] = np.asarray(len(sites), dtype=np.int32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    for h in handles.values():
        h.close()
    return result



def _facies_metrics(tau, wave, rho):
    """Compact observables that make the five classes readable."""
    tau = np.asarray(tau, float)
    wave = np.asarray(wave, float)
    rho = np.asarray(rho, float)
    after_s = tau >= 1.0
    if after_s.any():
        peak_i = int(np.nanargmax(np.where(after_s, wave, -np.inf)))
        t_peak = float(tau[peak_i] - 1.0)
        late = (tau >= 1.6) & (tau <= 2.4)
        late_frac = float(np.nanmean(wave[late]) / (np.nanmax(wave) + 1e-8)) if late.any() else 0.0
    else:
        t_peak, late_frac = 0.0, 0.0
    # rough multipath lobe count on ρ
    r = rho / (np.nanmax(rho) + 1e-8)
    peaks = 0
    for i in range(1, len(r) - 1):
        if r[i] > 0.35 and r[i] >= r[i - 1] and r[i] >= r[i + 1]:
            peaks += 1
    return t_peak, late_frac, peaks


def draw_a(fig, outer, means):
    """Five facies: envelope (left) and ρ(τ) (right), with quantitative tags."""
    gs = GridSpecFromSubplotSpec(
        len(SHAPE_ORDER), 2, subplot_spec=outer, hspace=0.12, wspace=0.08, width_ratios=[1.15, 0.85]
    )
    axes_l, axes_r = [], []
    for i, sh in enumerate(SHAPE_ORDER):
        ax = fig.add_subplot(gs[i, 0], sharex=axes_l[0] if axes_l else None)
        axr = fig.add_subplot(gs[i, 1], sharex=axes_r[0] if axes_r else None)
        axes_l.append(ax)
        axes_r.append(axr)
        clean(ax)
        clean(axr)
        m = means[sh]
        c = SHAPE_COLORS[sh]
        tau = np.asarray(m["tau"], float)
        wave = np.asarray(m["wave_mean"], float)
        wstd = np.asarray(m["wave_std"], float)
        rho = np.asarray(m["rho_mean"], float)
        rstd = np.asarray(m["rho_std"], float)
        rho_n = rho / (np.max(np.abs(rho)) + 1e-8)
        t_peak, late_frac, n_lobes = _facies_metrics(tau, wave, rho_n)

        ax.fill_between(tau, np.clip(wave - wstd, 0, None), wave + wstd, color=c, alpha=0.22, lw=0, zorder=1)
        ax.plot(tau, wave, color=c, lw=1.65, zorder=3, solid_capstyle="round")
        ax.axvline(0.0, color="#9CA3AF", lw=0.55, alpha=0.7, zorder=0)
        ax.axvline(1.0, color="#9CA3AF", lw=0.55, ls="--", alpha=0.55, zorder=0)
        ax.set_ylim(-0.05, 1.18)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(
            0.012, 0.92, f"{SHAPE_LABEL[sh]}  ·  n={m['n']}",
            transform=ax.transAxes, fontsize=6.5, color=c, fontweight="bold",
            ha="left", va="top", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.92),
        )
        ax.text(
            0.012, 0.12,
            f"t_peak−S={t_peak:.2f}   late={late_frac:.2f}",
            transform=ax.transAxes, fontsize=5.4, color=C_MUTED, ha="left", va="bottom",
        )

        axr.fill_between(tau, np.clip(rho_n - 0.55 * rstd, 0, None), rho_n + 0.55 * rstd, color=c, alpha=0.16, lw=0, zorder=1)
        axr.plot(tau, rho_n, color=c, lw=1.45, zorder=3)
        axr.axvline(0.0, color="#9CA3AF", lw=0.55, alpha=0.7, zorder=0)
        axr.axvline(1.0, color="#9CA3AF", lw=0.55, ls="--", alpha=0.55, zorder=0)
        axr.set_ylim(-0.05, 1.18)
        axr.set_yticks([])
        axr.spines["left"].set_visible(False)
        axr.text(0.98, 0.90, SHAPE_TAG[sh], transform=axr.transAxes, fontsize=5.5, color=C_MUTED, ha="right", va="top")
        axr.text(0.98, 0.12, f"ρ lobes≈{n_lobes}", transform=axr.transAxes, fontsize=5.4, color=C_MUTED, ha="right", va="bottom")

        if i == 0:
            ax.set_title("envelope", fontsize=6.5, color=C_MUTED, pad=2)
            axr.set_title(r"$\rho(\tau)$", fontsize=6.5, color=C_MUTED, pad=2)
            ax.text(0.0, 0.96, "P", transform=ax.get_xaxis_transform(), fontsize=6.2, color=C_MUTED, ha="center", va="top")
            ax.text(1.0, 0.96, "S", transform=ax.get_xaxis_transform(), fontsize=6.2, color=C_MUTED, ha="center", va="top")
        if i < len(SHAPE_ORDER) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(axr.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(r"$\tau$ (P=0, S=1)", fontsize=6.8, color=C_INK)
            axr.set_xlabel(r"$\tau$", fontsize=6.8, color=C_INK)
    axes_l[0].set_xlim(-0.25, 2.5)
    axes_r[0].set_xlim(-0.25, 2.5)
    return axes_l + axes_r



def draw_b(ax, cells, coast, faults, cax=None):
    # Fill the allocated cell height (match panel a); slight geo stretch is acceptable here.
    ax.set_xlim(-118.40, -115.40)
    ax.set_ylim(32.45, 34.70)
    ax.set_aspect("auto")
    clean(ax)
    ax.set_xlabel("longitude", fontsize=7)
    ax.set_ylabel("latitude", fontsize=7)
    for ring in coast:
        ax.fill(ring[:, 0], ring[:, 1], color="#F3F4F6", zorder=0)
        ax.plot(ring[:, 0], ring[:, 1], color="#D1D5DB", lw=0.5, zorder=1)
    for line in faults:
        ax.plot(line[:, 0], line[:, 1], color="#9CA3AF", lw=0.55, alpha=0.85, zorder=2)
    nrm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    sc = ax.scatter(
        cells["lon"], cells["lat"], c=cells["mean"],
        s=np.clip(cells["n"].to_numpy(float) * 0.5, 16, 150),
        cmap="coolwarm_r", norm=nrm, edgecolors="white", linewidths=0.25, zorder=3,
    )
    if cax is None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    else:
        cb = plt.colorbar(sc, cax=cax)
    cb.set_label(r"cell-mean $\beta_{\mathrm{res}}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.plot(-116.375, 33.375, marker="s", ms=10, mfc="none", mec=C_ABS, mew=1.6, zorder=5)
    ax.text(-116.20, 33.28, "Salton", color=C_ABS, fontsize=7.5, fontweight="bold", zorder=5)
    ax.plot(-116.375, 33.875, marker="s", ms=10, mfc="none", mec=C_RING, mew=1.6, zorder=5)
    ax.text(-116.55, 33.98, "E. Transverse R.", color=C_RING, fontsize=7.5, fontweight="bold",
            ha="right", zorder=5)
    ax.text(0.02, 0.03, "red = faster decay   blue = longer ringing", transform=ax.transAxes, fontsize=6.3, color=C_MUTED)


def draw_c(ax, ss, bysite):
    clean(ax)
    ax.axvline(0, color=C_LINE, lw=0.9, zorder=1)
    focus = [(2, "Salton", C_ABS), (5, "E. Transverse R.", C_RING)]
    y_cursor = 0.0
    yticks, ylabels = [], []
    for rank, lab, col in focus:
        s = bysite[bysite["rank"] == rank].sort_values("delta")
        ys = np.arange(len(s), dtype=float) + y_cursor
        ax.scatter(s["delta"], ys, s=22,
                   c=[C_ABS if v < 0 else C_RING for v in s["delta"]],
                   edgecolors="white", linewidths=0.3, zorder=3)
        for x, y in zip(s["delta"], ys):
            ax.plot([0, x], [y, y], color=C_LINE, lw=0.6, zorder=1)
        row = ss.loc[ss["rank"] == rank].iloc[0]
        med = float(row["site_delta_med"])
        n = int(row["n_sites_paired"])
        p = float(row["mw_p"])
        ax.plot([med, med], [ys.min() - 0.45, ys.max() + 0.45], color=col, lw=1.8, zorder=2)
        ax.text(med, ys.max() + 0.7, f"med={med:+.3f}", color=col, fontsize=6.2, ha="center", va="bottom", fontweight="bold")
        yticks.append(float(ys.mean()))
        ylabels.append(f"{lab}\nn={n}, p≈{p:.1g}")
        y_cursor = float(ys.max()) + 2.6
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel(r"same-station $\Delta\beta$  (in-cell − other paths)", fontsize=7.5)
    ax.set_xlim(-0.09, 0.09)
    ax.set_ylim(-1.2, y_cursor - 1.2)
    ax.text(0.98, 0.04, "negative = faster in-cell", transform=ax.transAxes, fontsize=6.4, color=C_ABS, ha="right")



def draw_d(fig, outer, stacks, ss):
    """In vs out stacks with residual curve and late-coda contrast."""
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, hspace=0.34, height_ratios=[1, 1])
    axes = []
    t = np.asarray(stacks["t"], float)
    for i, cell in enumerate(CELLS):
        cell_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i], height_ratios=[2.4, 1.0], hspace=0.08)
        ax = fig.add_subplot(cell_gs[0])
        axd = fig.add_subplot(cell_gs[1], sharex=ax)
        axes.extend([ax, axd])
        clean(ax)
        clean(axd)
        st = stacks[f"rank{cell['rank']}"]
        col = cell["color"]
        ax.axvspan(8.0, 18.0, color="#F3F4F6", zorder=0, lw=0)
        ax.fill_between(t, st["out_p25"], st["out_p75"], color="#9CA3AF", alpha=0.20, lw=0, zorder=1)
        ax.fill_between(t, st["in_p25"], st["in_p75"], color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(t, st["out_mean"], color=C_MUTED, lw=1.5, zorder=3, label="out-of-cell")
        ax.plot(t, st["in_mean"], color=col, lw=1.9, zorder=4, label="in-cell")
        ax.axvline(0.0, color="#9CA3AF", lw=0.7, alpha=0.8, zorder=0)
        ax.set_xlim(-1.0, 24)
        ax.set_ylim(0.0, 1.05)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_ylabel("envelope", fontsize=7)

        # residual: in - out
        diff = st["in_mean"] - st["out_mean"]
        axd.axhline(0.0, color="#9CA3AF", lw=0.8, zorder=0)
        axd.axvspan(8.0, 18.0, color="#F3F4F6", zorder=0, lw=0)
        axd.fill_between(t, 0, diff, where=diff >= 0, color=C_RING, alpha=0.28, lw=0, interpolate=True)
        axd.fill_between(t, 0, diff, where=diff < 0, color=C_ABS, alpha=0.28, lw=0, interpolate=True)
        axd.plot(t, diff, color=col, lw=1.7, zorder=3)
        # zoom residual to the data range so the late-coda sign is obvious
        span = float(np.nanmax(np.abs(diff)))
        axd.set_ylim(-1.15 * max(span, 0.12), 1.15 * max(span, 0.12))
        axd.set_ylabel(r"in−out", fontsize=6.5)
        if i == 1:
            axd.set_xlabel("time after S (s)", fontsize=7)
        else:
            plt.setp(axd.get_xticklabels(), visible=False)

        row = ss.loc[ss["rank"] == cell["rank"]].iloc[0]
        med = float(row["site_delta_med"])
        late = (t >= 8.0) & (t <= 18.0)
        in_late = float(np.nanmean(st["in_mean"][late]))
        out_late = float(np.nanmean(st["out_mean"][late]))
        ratio = in_late / (out_late + 1e-9)
        # simple e-folding from 2–12 s
        win = (t >= 2.0) & (t <= 12.0)
        def _efold(y):
            yy = np.clip(np.asarray(y)[win], 1e-4, None)
            tt = t[win]
            slope = np.polyfit(tt, np.log(yy), 1)[0]
            return -1.0 / slope if slope < 0 else np.nan
        tin = _efold(st["in_mean"])
        tout = _efold(st["out_mean"])

        ax.text(
            0.01, 0.96,
            f"{cell['short']}   Δβ={med:+.3f}   ·   {cell['note']}",
            transform=ax.transAxes, fontsize=7.0, color=col, fontweight="bold", ha="left", va="top",
        )
        ax.text(
            0.01, 0.72,
            f"n_in={st['n_in']}, n_out={st['n_out']} ({st['n_sites']} sta.)\n"
            f"late 8–18 s in/out={ratio:.2f}   τ_e in/out={tin:.1f}/{tout:.1f}s",
            transform=ax.transAxes, fontsize=5.7, color=C_MUTED, ha="left", va="top", linespacing=1.25,
        )
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=5.6, handlelength=1.3)
        axd.text(0.99, 0.90, "in − out", transform=axd.transAxes, fontsize=5.5, color=C_MUTED, ha="right", va="top")
    return axes



def main() -> None:
    args = parse_args()
    style()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    socal = Path(args.socal_dir)
    if not socal.is_absolute():
        socal = _REPO / socal
    means_path = Path(args.means)
    if not means_path.is_absolute():
        means_path = _REPO / means_path
    panel_csv = Path(args.panel)
    if not panel_csv.is_absolute():
        panel_csv = _REPO / panel_csv

    cells = pd.read_csv(socal / "grid_beta_resid.csv")
    ss = pd.read_csv(socal / "same_station_validation.csv")
    bysite = pd.read_csv(socal / "same_station_by_site.csv")
    means = load_means(means_path)

    coast = load_lines(_REPO / "docs/figures/geo/ne_110m_land.geojson")
    faults = load_lines(_REPO / "docs/figures/geo/qfaults_socal.geojson") + load_lines(
        _REPO / "docs/figures/geo/qfaults_socal_offshore.geojson"
    )

    cache = out_dir / "fig2_envelope_stacks.npz"
    stacks = build_or_load_stacks(
        cache,
        traces=socal / "traces_with_structure.csv",
        panel_csv=panel_csv,
        bysite=socal / "same_station_by_site.csv",
        stead_dir=Path(args.stead_dir),
        n_stations=args.n_stations,
        max_per_arm=args.max_per_arm,
        rebuild=args.rebuild_stacks,
    )


    fig = plt.figure(figsize=(8.8, 9.6), dpi=150)
    # title | top panels | spacer | title | bottom panels
    gs = GridSpec(
        5, 2, figure=fig,
        height_ratios=[0.048, 1.65, 0.30, 0.048, 1.05],
        width_ratios=[1.0, 1.20],
        left=0.12, right=0.92, top=0.96, bottom=0.07,
        wspace=0.52, hspace=0.04,
    )

    ax_ta = fig.add_subplot(gs[0, 0]); ax_ta.axis("off")
    ax_tb = fig.add_subplot(gs[0, 1]); ax_tb.axis("off")
    ax_tc = fig.add_subplot(gs[3, 0]); ax_tc.axis("off")
    ax_td = fig.add_subplot(gs[3, 1]); ax_td.axis("off")

    axes_a = draw_a(fig, gs[1, 0], means)
    # Nested map + colorbar so the map box can fill the full row height.
    gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], width_ratios=[1.0, 0.045], wspace=0.08)
    ax_b = fig.add_subplot(gs_b[0, 0])
    cax_b = fig.add_subplot(gs_b[0, 1])
    draw_b(ax_b, cells, coast, faults, cax=cax_b)
    ax_c = fig.add_subplot(gs[4, 0])
    draw_c(ax_c, ss, bysite)
    axes_d = draw_d(fig, gs[4, 1], stacks, ss)

    def _hdr(ax, letter, title):
        ax.text(0.0, 0.15, letter, fontsize=11, fontweight="bold", color=C_INK, va="bottom", ha="left", transform=ax.transAxes)
        ax.text(0.06, 0.15, title, fontsize=8.5, color=C_INK, va="bottom", ha="left", transform=ax.transAxes)

    _hdr(ax_ta, "a", "Five frozen coda facies (class means)")
    _hdr(ax_tb, "b", r"SoCal path-midpoint $\beta_{\mathrm{res}}$")
    _hdr(ax_tc, "c", r"Same-station $\Delta\beta$")
    _hdr(ax_td, "d", "S-aligned in-cell vs out-of-cell stacks")

    png = out_dir / "fig2_seismic.png"
    pdf = out_dir / "fig2_seismic.pdf"
    fig.savefig(png, dpi=args.dpi, pad_inches=0.15)
    fig.savefig(pdf, pad_inches=0.15)
    plt.close(fig)
    print({"png": str(png), "pdf": str(pdf), "stacks": str(cache)})


if __name__ == "__main__":
    main()
