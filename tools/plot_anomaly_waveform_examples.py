#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-station waveform envelopes for surviving structure-anomaly cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEAD_DIR

C_INK = "#1C1917"
C_MUTED = "#78716C"
BG = "#FFFEF9"
GRID = 0.25

CELLS = [
    {"rank": 2, "lon": -116.375, "lat": 33.375, "title": "#2 Salton Trough / S.SAF–Imperial", "color": "#B91C1C"},
    {"rank": 5, "lon": -116.375, "lat": 33.875, "title": "#5 E. Transverse Ranges / LSB Mts", "color": "#1D4ED8"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="outputs/structure_residual_socal/traces_with_structure.csv")
    p.add_argument("--panel", default="outputs/shape_labels_expanded/socal/panel_selection.csv")
    p.add_argument("--by-site", default="outputs/structure_residual_socal/same_station_by_site.csv")
    p.add_argument("--stead-dir", default=str(STEAD_DIR))
    p.add_argument("--n-stations", type=int, default=3)
    p.add_argument("--output", default="outputs/structure_residual_socal/anomaly_waveform_examples.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _cell_key(lon, lat):
    gx = np.floor(lon / GRID) * GRID + 0.5 * GRID
    gy = np.floor(lat / GRID) * GRID + 0.5 * GRID
    return round(float(gx), 6), round(float(gy), 6)


def _env(wave: np.ndarray) -> np.ndarray:
    e = np.sqrt(np.mean(np.square(wave.astype(np.float64)), axis=1) + 1e-12)
    e = e / (np.max(e) + 1e-12)
    # light smooth
    k = np.ones(21) / 21.0
    return np.convolve(e, k, mode="same")


def _lookup_chunk(panel: pd.DataFrame) -> dict[str, int]:
    return dict(zip(panel["trace_name"].astype(str), panel["chunk"].astype(int)))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.traces)
    panel = pd.read_csv(args.panel, usecols=["trace_name", "chunk"])
    bysite = pd.read_csv(args.by_site)
    chunk_of = _lookup_chunk(panel)
    df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])

    fig, axes = plt.subplots(len(CELLS), args.n_stations, figsize=(11.0, 6.6), facecolor=BG, sharex=True, sharey=True)
    if args.n_stations == 1:
        axes = np.array(axes).reshape(len(CELLS), 1)

    handles: dict[int, h5py.File] = {}
    stead = Path(args.stead_dir)

    def h5(chunk: int) -> h5py.File:
        if chunk not in handles:
            handles[chunk] = h5py.File(stead / f"chunk{chunk}_eofextract" / f"chunk{chunk}.hdf5", "r")
        return handles[chunk]

    t = np.linspace(0, 60.0, 6000)
    for i, cell in enumerate(CELLS):
        sites = (
            bysite[bysite["rank"] == cell["rank"]]
            .assign(absd=lambda d: np.abs(d["delta"]))
            .sort_values("absd", ascending=False)
        )
        sites = sites[sites["n_in"] >= 3].head(args.n_stations)
        in_m = np.isclose(df["gx"], cell["lon"]) & np.isclose(df["gy"], cell["lat"])
        buf = 0.5 * GRID
        near = (np.abs(df["gx"] - cell["lon"]) <= buf + 1e-9) & (np.abs(df["gy"] - cell["lat"]) <= buf + 1e-9)
        for j in range(args.n_stations):
            ax = axes[i, j]
            ax.set_facecolor(BG)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if j >= len(sites):
                ax.axis("off")
                continue
            site = str(sites.iloc[j]["site"])
            sub_in = df[in_m & (df["site"] == site)]
            sub_out = df[(~near) & (df["site"] == site)]
            shown_in = shown_out = 0
            for _, row in sub_in.head(8).iterrows():
                name = str(row["trace_name"])
                ch = chunk_of.get(name)
                if ch is None:
                    continue
                try:
                    w = h5(int(ch))["data"][name][()]
                except Exception:
                    continue
                ax.plot(t, _env(w), color=cell["color"], lw=0.8, alpha=0.55)
                shown_in += 1
            for _, row in sub_out.head(8).iterrows():
                name = str(row["trace_name"])
                ch = chunk_of.get(name)
                if ch is None:
                    continue
                try:
                    w = h5(int(ch))["data"][name][()]
                except Exception:
                    continue
                ax.plot(t, _env(w), color="#6B7280", lw=0.7, alpha=0.35)
                shown_out += 1
            ax.set_xlim(0, 40)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{site}\nΔβ={sites.iloc[j]['delta']:+.3f}", fontsize=8, color=C_INK)
            else:
                ax.set_title(f"{site}\nΔβ={sites.iloc[j]['delta']:+.3f}", fontsize=8, color=C_INK)
            if j == 0:
                ax.set_ylabel(f"{cell['title']}\nenvelope", fontsize=7.5, color=C_INK)
            ax.text(0.98, 0.92, f"in={shown_in}  ctrl={shown_out}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=C_MUTED)
            if i == len(CELLS) - 1:
                ax.set_xlabel("Time (s)", fontsize=8, color=C_INK)

    fig.suptitle(
        "Same-station envelopes: path midpoint inside anomaly cell (color) vs elsewhere (grey)",
        fontsize=11,
        color=C_INK,
        y=0.99,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.94))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi, facecolor=BG)
    plt.close(fig)
    for h in handles.values():
        h.close()
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
