#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent shallow-structure overlay: path-midpoint Vs30 proxy.

Station FE already absorbs site Vs30. This script asks whether cells #2/#5
are just 'paths that cross basins', using a Wald & Allen (2007) slope-based
Vs30 proxy evaluated at the *path midpoint*.

Elevation comes from local SRTM1 (Mapzen/AWS skadi) tiles — no rate-limited API.

Example
-------
  PYTHONPATH=. python tools/overlay_path_vs30_proxy.py
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_INK = "#1C1917"
C_MUTED = "#78716C"
BG = "#FFFEF9"
GRID = 0.25
CELLS = {
    2: (-116.375, 33.375, "Salton / S.SAF–Imperial"),
    5: (-116.375, 33.875, "E. Transverse Ranges"),
}

_SLOPE = np.array([3.0e-4, 3.5e-3, 1.0e-2, 1.8e-2, 5.0e-2, 1.0e-1, 1.4e-1])
_VS30 = np.array([180.0, 240.0, 300.0, 360.0, 490.0, 620.0, 760.0])
SKADI = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="outputs/structure_residual_socal/traces_with_structure.csv")
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--dem-dir", default="data/external_tomo/srtm1_skadi")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _cell_key(lon, lat, grid=GRID):
    gx = np.floor(lon / grid) * grid + 0.5 * grid
    gy = np.floor(lat / grid) * grid + 0.5 * grid
    return round(float(gx), 6), round(float(gy), 6)


def vs30_from_slope(slope_mm: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(slope_mm, dtype=float), _SLOPE[0], _SLOPE[-1])
    return np.interp(np.log10(s), np.log10(_SLOPE), _VS30)


def tile_name(lat: float, lon: float) -> str:
    la = int(math.floor(lat))
    lo = int(math.floor(lon))
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"


def ensure_tile(name: str, dem_dir: Path) -> Path:
    dem_dir.mkdir(parents=True, exist_ok=True)
    out = dem_dir / f"{name}.hgt"
    if out.is_file() and out.stat().st_size >= 3601 * 3601 * 2:
        return out
    folder = name[:3]
    url = f"{SKADI}/{folder}/{name}.hgt.gz"
    print(f"[dem] fetch {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "HNF/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = gzip.decompress(resp.read())
    if len(raw) != 3601 * 3601 * 2:
        raise RuntimeError(f"{name}: unexpected hgt size {len(raw)}")
    out.write_bytes(raw)
    return out


class SrtmMosaic:
    def __init__(self, dem_dir: Path):
        self.dem_dir = dem_dir
        self.cache: dict[str, np.ndarray] = {}

    def _arr(self, name: str) -> np.ndarray:
        if name not in self.cache:
            path = ensure_tile(name, self.dem_dir)
            self.cache[name] = np.fromfile(path, dtype=">i2").reshape(3601, 3601)
        return self.cache[name]

    def sample(self, lon: float, lat: float) -> float:
        name = tile_name(lat, lon)
        arr = self._arr(name)
        la0 = math.floor(lat)
        lo0 = math.floor(lon)
        # pixel (0,0) is NW corner of tile = (la0+1, lo0)
        row = (la0 + 1.0 - lat) * 3600.0
        col = (lon - lo0) * 3600.0
        r0 = int(np.clip(math.floor(row), 0, 3599))
        c0 = int(np.clip(math.floor(col), 0, 3599))
        dr, dc = row - r0, col - c0
        vals = np.array(
            [arr[r0, c0], arr[r0, c0 + 1], arr[r0 + 1, c0], arr[r0 + 1, c0 + 1]],
            dtype=float,
        )
        vals[vals < -32000] = np.nan
        if np.isnan(vals).all():
            return float("nan")
        w = np.array([(1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc])
        m = np.isfinite(vals)
        return float(np.sum(vals[m] * w[m]) / np.sum(w[m]))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.traces)
    df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])
    mos = SrtmMosaic(Path(args.dem_dir))

    # unique rounded nodes at ~250 m to keep tile IO modest
    rnd = 0.0025
    df["qlon"] = np.round(df["path_mid_lon"] / rnd) * rnd
    df["qlat"] = np.round(df["path_mid_lat"] / rnd) * rnd
    nodes = df[["qlon", "qlat"]].drop_duplicates().reset_index(drop=True)
    print(f"[vs30] sampling {len(nodes)} path nodes on SRTM1", flush=True)

    dlt_deg = 0.0025  # ~250 m stencil
    elevs, slopes, vs30s = [], [], []
    for i, (lon, lat) in enumerate(nodes.itertuples(index=False)):
        c = mos.sample(lon, lat)
        e = mos.sample(lon + dlt_deg, lat)
        w = mos.sample(lon - dlt_deg, lat)
        n = mos.sample(lon, lat + dlt_deg)
        s = mos.sample(lon, lat - dlt_deg)
        dx = dlt_deg * 111320.0 * math.cos(math.radians(lat))
        dy = dlt_deg * 111320.0
        dzdx = (e - w) / (2.0 * dx) if np.isfinite(e) and np.isfinite(w) and dx > 0 else np.nan
        dzdy = (n - s) / (2.0 * dy) if np.isfinite(n) and np.isfinite(s) else np.nan
        slope = float(math.hypot(dzdx, dzdy)) if np.isfinite(dzdx) and np.isfinite(dzdy) else np.nan
        elevs.append(c)
        slopes.append(slope)
        vs30s.append(float(vs30_from_slope([max(slope, _SLOPE[0])])[0]) if np.isfinite(slope) else np.nan)
        if (i + 1) % 2000 == 0:
            print(f"[vs30] {i+1}/{len(nodes)}", flush=True)
    nodes["elev_m"] = elevs
    nodes["slope"] = slopes
    nodes["vs30"] = vs30s
    work = df.merge(nodes, on=["qlon", "qlat"], how="left").dropna(subset=["beta_resid", "vs30"])
    print(f"[vs30] traces with proxy={len(work)}", flush=True)

    x = np.log10(work["vs30"].to_numpy(float))
    y = work["beta_resid"].to_numpy(float)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    work["beta_after_vs30"] = y - A @ coef
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 5 else float("nan")

    rows = []
    for rank, (lon, lat, name) in CELLS.items():
        m = np.isclose(work["gx"], lon) & np.isclose(work["gy"], lat)
        sub, ctrl = work[m], work[~m]
        rows.append(
            {
                "rank": rank,
                "name": name,
                "n_cell": int(len(sub)),
                "vs30_cell": float(sub["vs30"].median()),
                "vs30_ctrl": float(ctrl["vs30"].median()),
                "elev_cell": float(sub["elev_m"].median()),
                "elev_ctrl": float(ctrl["elev_m"].median()),
                "delta_beta": float(sub["beta_resid"].median() - ctrl["beta_resid"].median()),
                "delta_beta_after_vs30": float(
                    sub["beta_after_vs30"].median() - ctrl["beta_after_vs30"].median()
                ),
            }
        )
    summ = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work[
        ["trace_name", "site", "gx", "gy", "beta_resid", "vs30", "elev_m", "slope", "beta_after_vs30"]
    ].to_csv(out / "beta_vs_path_vs30.csv", index=False)
    summ.to_csv(out / "beta_vs_path_vs30_cells.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    sc = axes[0].scatter(
        work["path_mid_lon"],
        work["path_mid_lat"],
        c=work["vs30"],
        s=5,
        cmap="viridis",
        vmin=180,
        vmax=760,
        linewidths=0,
        rasterized=True,
    )
    for rank, (lon, lat, _) in CELLS.items():
        axes[0].plot(lon, lat, marker="s", ms=9, mfc="none", mec="red" if rank == 2 else "white", mew=1.4)
        axes[0].text(lon + 0.05, lat + 0.05, f"#{rank}", color="k", fontsize=8, fontweight="bold")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Path-midpoint Vs30 proxy (Wald–Allen / SRTM)", fontsize=9, color=C_INK)
    axes[0].set_xlabel("lon", fontsize=8)
    axes[0].set_ylabel("lat", fontsize=8)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("Vs30 (m/s)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    axes[1].scatter(work["vs30"], work["beta_resid"], s=6, c="#9CA3AF", alpha=0.35, linewidths=0, rasterized=True)
    xx = np.linspace(np.nanpercentile(work["vs30"], 2), np.nanpercentile(work["vs30"], 98), 40)
    axes[1].plot(xx, coef[0] + coef[1] * np.log10(xx), color=C_INK, lw=1.2)
    axes[1].set_xlabel("path-midpoint Vs30 proxy (m/s)", fontsize=8)
    axes[1].set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=8)
    axes[1].set_title(f"β_res vs path Vs30\nPearson r={r:.2f}", fontsize=9, color=C_INK)

    xs = np.arange(len(summ))
    axes[2].bar(xs - 0.18, summ["delta_beta"], width=0.36, color="#6B7280", label="before Vs30")
    axes[2].bar(xs + 0.18, summ["delta_beta_after_vs30"], width=0.36, color="#1D4ED8", label="after Vs30")
    axes[2].axhline(0, color=C_MUTED, lw=0.7, ls="--")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(
        [f"#{int(r.rank)}\n{r.name.split('/')[0].strip()[:16]}" for r in summ.itertuples()], fontsize=7
    )
    axes[2].set_ylabel(r"cell − rest median $\Delta\beta$", fontsize=8)
    axes[2].set_title("Do #2/#5 survive path Vs30?", fontsize=9, color=C_INK)
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "beta_vs_path_vs30.png", dpi=args.dpi, facecolor=BG)
    fig.savefig(out / "beta_vs_path_vs30.pdf", dpi=args.dpi, facecolor=BG)
    plt.close(fig)

    report = {
        "n": int(len(work)),
        "pearson_beta_vs_log_vs30": r,
        "coef": [float(coef[0]), float(coef[1])],
        "cells": summ.to_dict(orient="records"),
        "note": "Vs30 is Wald & Allen 2007 active-tectonic slope proxy at path midpoint from SRTM1; station FE already removed site Vs30.",
    }
    (out / "beta_vs_path_vs30_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] {out / 'beta_vs_path_vs30.png'}", flush=True)


if __name__ == "__main__":
    main()
