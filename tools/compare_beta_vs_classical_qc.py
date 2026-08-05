#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare interpretable β_res residuals against a classical single-parameter Qc proxy.

For each trace, estimate a classical coda quality factor from the observed
envelope decay after S (log-energy vs time), then ask:

  1. Does cell #2 / #5 still stand out after regressing β_res on classical Qc?
  2. Within a narrow Qc band, do multipath vs slow_coda still separate?

This is the "more than a single Q" test required for a Nature-family claim
when a published tomography cube is not locally available.

Example
-------
  PYTHONPATH=. python tools/compare_beta_vs_classical_qc.py
"""

from __future__ import annotations

import argparse
import json
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
CELLS = {
    2: (-116.375, 33.375, "#B91C1C", "Salton / S.SAF–Imperial"),
    5: (-116.375, 33.875, "#1D4ED8", "E. Transverse Ranges"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="outputs/structure_residual_socal/traces_with_structure.csv")
    p.add_argument("--panel", default="outputs/shape_labels_expanded/socal/panel_selection.csv")
    p.add_argument("--stead-dir", default=str(STEAD_DIR))
    p.add_argument("--max-traces", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _env(wave: np.ndarray) -> np.ndarray:
    e = np.mean(np.square(wave.astype(np.float64)), axis=1)
    return np.convolve(e, np.ones(21) / 21.0, mode="same")


def classical_qc_slope(wave: np.ndarray, s_frac: float | None = None) -> float:
    """Return d log10(E) / dt over an 8 s window after an energy-based S proxy.

    More negative => faster decay => lower apparent Q.
    """
    e = _env(wave)
    n = e.size
    t = np.linspace(0.0, 60.0, n)
    # crude S proxy: first time after 20% of max energy that energy falls below 55% of peak after the peak
    peak = int(np.argmax(e))
    s_idx = peak
    for i in range(peak, min(n - 1, peak + 800)):
        if e[i] < 0.55 * e[peak]:
            s_idx = i
            break
    lo = t[s_idx]
    hi = min(60.0, lo + 8.0)
    m = (t >= lo) & (t <= hi)
    if m.sum() < 20:
        return float("nan")
    y = np.log10(e[m] + 1e-12)
    x = t[m]
    if np.std(x) < 1e-6:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _cell_key(lon, lat):
    gx = np.floor(lon / GRID) * GRID + 0.5 * GRID
    gy = np.floor(lat / GRID) * GRID + 0.5 * GRID
    return round(float(gx), 6), round(float(gy), 6)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.traces)
    panel = pd.read_csv(args.panel, usecols=["trace_name", "chunk", "p_arrival_sample", "s_arrival_sample"] if False else ["trace_name", "chunk"])
    chunk_of = dict(zip(panel["trace_name"].astype(str), panel["chunk"].astype(int)))
    rng = np.random.default_rng(args.seed)
    # prefer including all traces that fall in target cells, then fill
    df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])
    must = pd.Series(False, index=df.index)
    for lon, lat, _, _ in CELLS.values():
        must |= np.isclose(df["gx"], lon) & np.isclose(df["gy"], lat)
    idx_must = df.index[must].to_numpy()
    idx_rest = df.index[~must].to_numpy()
    need = max(args.max_traces - len(idx_must), 0)
    if need > 0 and len(idx_rest) > 0:
        take = rng.choice(idx_rest, size=min(need, len(idx_rest)), replace=False)
        sel = np.concatenate([idx_must, take])
    else:
        sel = idx_must
    work = df.loc[sel].copy()
    print(f"[qc] computing classical decay on {len(work)} traces", flush=True)

    handles: dict[int, h5py.File] = {}
    stead = Path(args.stead_dir)

    def h5(c):
        if c not in handles:
            handles[c] = h5py.File(stead / f"chunk{c}_eofextract" / f"chunk{c}.hdf5", "r")
        return handles[c]

    slopes = []
    for i, row in enumerate(work.itertuples()):
        name = str(row.trace_name)
        ch = chunk_of.get(name)
        if ch is None:
            slopes.append(np.nan)
            continue
        try:
            w = h5(int(ch))["data"][name][()]
            slopes.append(classical_qc_slope(w))
        except Exception:
            slopes.append(np.nan)
        if (i + 1) % 500 == 0:
            print(f"[qc] {i+1}/{len(work)}", flush=True)
    work["qc_slope"] = slopes
    work = work.dropna(subset=["qc_slope", "beta_resid"])
    print(f"[qc] kept {len(work)} with finite classical slope", flush=True)
    for h in handles.values():
        h.close()

    # residualize beta on classical qc (+ dist already in beta_resid, but qc still correlated)
    x = work["qc_slope"].to_numpy(float)
    y = work["beta_resid"].to_numpy(float)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    work["beta_after_qc"] = y - A @ coef
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 5 else float("nan")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work[["trace_name", "site", "shape", "gx", "gy", "beta_resid", "qc_slope", "beta_after_qc", "p_mp_resid", "p_sc_resid"]].to_csv(
        out / "beta_vs_classical_qc.csv", index=False
    )

    # cell summaries before/after removing classical Qc
    rows = []
    for rank, (lon, lat, col, name) in CELLS.items():
        m = np.isclose(work["gx"], lon) & np.isclose(work["gy"], lat)
        sub = work[m]
        ctrl = work[~m]
        if len(sub) < 10:
            continue
        rows.append(
            {
                "rank": rank,
                "name": name,
                "n_cell": int(len(sub)),
                "n_ctrl": int(len(ctrl)),
                "beta_resid_cell": float(sub["beta_resid"].median()),
                "beta_resid_ctrl": float(ctrl["beta_resid"].median()),
                "beta_after_qc_cell": float(sub["beta_after_qc"].median()),
                "beta_after_qc_ctrl": float(ctrl["beta_after_qc"].median()),
                "qc_slope_cell": float(sub["qc_slope"].median()),
                "qc_slope_ctrl": float(ctrl["qc_slope"].median()),
                "delta_beta": float(sub["beta_resid"].median() - ctrl["beta_resid"].median()),
                "delta_beta_after_qc": float(sub["beta_after_qc"].median() - ctrl["beta_after_qc"].median()),
                "delta_qc": float(sub["qc_slope"].median() - ctrl["qc_slope"].median()),
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "beta_vs_classical_qc_cells.csv", index=False)

    # within Qc terciles: multipath vs slow_coda beta
    work["qc_bin"] = pd.qcut(work["qc_slope"], 3, labels=["fastQ", "midQ", "slowQ"])
    sep_rows = []
    for b, sub in work.groupby("qc_bin"):
        a = sub.loc[sub["shape"] == "multipath", "beta_resid"].to_numpy(float)
        c = sub.loc[sub["shape"] == "slow_coda", "beta_resid"].to_numpy(float)
        if len(a) < 8 or len(c) < 8:
            continue
        sep_rows.append(
            {
                "qc_bin": str(b),
                "n_mp": int(len(a)),
                "n_sc": int(len(c)),
                "beta_mp": float(np.nanmedian(a)),
                "beta_sc": float(np.nanmedian(c)),
                "delta": float(np.nanmedian(c) - np.nanmedian(a)),
            }
        )
    sep = pd.DataFrame(sep_rows)
    sep.to_csv(out / "shape_separation_within_qc_bins.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].scatter(work["qc_slope"], work["beta_resid"], s=6, c="#9CA3AF", alpha=0.35, linewidths=0, rasterized=True)
    xx = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 50)
    axes[0].plot(xx, coef[0] + coef[1] * xx, color=C_INK, lw=1.2)
    axes[0].set_xlabel("Classical coda decay slope  (more negative = faster)", fontsize=8)
    axes[0].set_ylabel(r"Structure residual $\beta_{\mathrm{res}}$", fontsize=8)
    axes[0].set_title(f"β_res vs classical Qc proxy\nPearson r={r:.2f}", fontsize=9, color=C_INK)

    labels, d0, d1 = [], [], []
    for rec in summ.itertuples():
        labels.append(f"#{int(rec.rank)}\n{rec.name.split('/')[0].strip()[:16]}")
        d0.append(rec.delta_beta)
        d1.append(rec.delta_beta_after_qc)
    xs = np.arange(len(labels))
    axes[1].bar(xs - 0.18, d0, width=0.36, color="#6B7280", label="before removing Qc")
    axes[1].bar(xs + 0.18, d1, width=0.36, color="#B91C1C", label="after removing Qc")
    axes[1].axhline(0, color=C_MUTED, lw=0.7, ls="--")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel(r"cell − rest  median $\Delta\beta$", fontsize=8)
    axes[1].set_title("Do #2/#5 survive classical Qc?", fontsize=9, color=C_INK)
    axes[1].legend(fontsize=7, frameon=False)

    if not sep.empty:
        axes[2].bar(np.arange(len(sep)) - 0.18, sep["beta_mp"], width=0.36, color="#2E8B57", label="multipath")
        axes[2].bar(np.arange(len(sep)) + 0.18, sep["beta_sc"], width=0.36, color="#B85C38", label="slow_coda")
        axes[2].set_xticks(np.arange(len(sep)))
        axes[2].set_xticklabels(sep["qc_bin"], fontsize=8)
        axes[2].set_ylabel(r"median $\beta_{\mathrm{res}}$", fontsize=8)
        axes[2].set_title("Shape split inside Qc terciles", fontsize=9, color=C_INK)
        axes[2].legend(fontsize=7, frameon=False)
        axes[2].axhline(0, color=C_MUTED, lw=0.7, ls="--")
    fig.tight_layout()
    fig.savefig(out / "beta_vs_classical_qc.png", dpi=args.dpi, facecolor=BG)
    fig.savefig(out / "beta_vs_classical_qc.pdf", dpi=args.dpi, facecolor=BG)
    plt.close(fig)

    report = {
        "n": int(len(work)),
        "pearson_beta_vs_qc": r,
        "qc_coef_on_beta_resid": [float(coef[0]), float(coef[1])],
        "cells": summ.to_dict(orient="records"),
        "within_qc_shape_split": sep.to_dict(orient="records"),
    }
    (out / "beta_vs_classical_qc_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] {out / 'beta_vs_classical_qc.png'}", flush=True)


if __name__ == "__main__":
    main()
