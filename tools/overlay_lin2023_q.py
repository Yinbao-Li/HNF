#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay Lin & Jordan 2023 SCAT QS/QP on facies residuals.

Gate A (attenuation): do #2/#5 retrace published Q provinces, or split a Q band?

Example
-------
  PYTHONPATH=. python tools/overlay_lin2023_q.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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
    p.add_argument("--qs", default="data/external_tomo/SouthernCalifornia_QSmodel.txt")
    p.add_argument("--qp", default="data/external_tomo/SouthernCalifornia_QPmodel.txt")
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _cell_key(lon, lat, grid=GRID):
    gx = np.floor(lon / grid) * grid + 0.5 * grid
    gy = np.floor(lat / grid) * grid + 0.5 * grid
    return round(float(gx), 6), round(float(gy), 6)


def load_q(path: Path):
    arr = np.loadtxt(path, comments="#")
    return {
        "lon": arr[:, 0],
        "lat": arr[:, 1],
        "z": arr[:, 2],
        "qavg": arr[:, 3],
        "q": arr[:, 4],
        "qpct": arr[:, 5],
        "res": arr[:, 6],
    }


class LayeredQ:
    def __init__(self, tab: dict, resolved_only: bool = True):
        self.depths = np.array(sorted(np.unique(np.round(tab["z"], 3))))
        self.lin: dict[float, LinearNDInterpolator] = {}
        self.nn: dict[float, NearestNDInterpolator] = {}
        self.xy_res: dict[float, np.ndarray] = {}
        self.q_res: dict[float, np.ndarray] = {}
        for z in self.depths:
            m = np.isclose(tab["z"], z)
            if resolved_only:
                m = m & (tab["res"] >= 0.5)
            xy = np.column_stack([tab["lon"][m], tab["lat"][m]])
            q = tab["q"][m]
            if len(xy) < 8:
                continue
            self.lin[float(z)] = LinearNDInterpolator(xy, q)
            self.nn[float(z)] = NearestNDInterpolator(xy, q)
            self.xy_res[float(z)] = xy
            self.q_res[float(z)] = q

    def sample(self, lon: float, lat: float, z: float) -> float:
        if not self.lin:
            return float("nan")
        zs = np.array(sorted(self.lin))
        # interpolate between nearest depth layers
        if z <= zs[0]:
            return self._xy(zs[0], lon, lat)
        if z >= zs[-1]:
            return self._xy(zs[-1], lon, lat)
        i = int(np.searchsorted(zs, z))
        z0, z1 = float(zs[i - 1]), float(zs[i])
        v0, v1 = self._xy(z0, lon, lat), self._xy(z1, lon, lat)
        if not np.isfinite(v0):
            return v1
        if not np.isfinite(v1):
            return v0
        t = (z - z0) / max(z1 - z0, 1e-9)
        return float((1 - t) * v0 + t * v1)

    def _xy(self, z: float, lon: float, lat: float) -> float:
        v = float(self.lin[z](lon, lat))
        if np.isfinite(v):
            return v
        # only accept nearest if reasonably close (< 25 km)
        xy = self.xy_res[z]
        d2 = (xy[:, 0] - lon) ** 2 + (xy[:, 1] - lat) ** 2
        j = int(np.argmin(d2))
        if d2[j] > (0.25**2):
            return float("nan")
        return float(self.q_res[z][j])

    def column_mean(self, lon: float, lat: float, z_lo: float, z_hi: float) -> float:
        zs = [z for z in self.lin if z_lo - 1e-6 <= z <= z_hi + 1e-6]
        vals = [self._xy(z, lon, lat) for z in zs]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def map_at(self, z: float) -> tuple[np.ndarray, np.ndarray]:
        zs = np.array(sorted(self.lin))
        zz = float(zs[np.argmin(np.abs(zs - z))])
        return self.xy_res[zz], self.q_res[zz]


def main() -> None:
    args = parse_args()
    qs = LayeredQ(load_q(Path(args.qs)), resolved_only=True)
    qp = LayeredQ(load_q(Path(args.qp)), resolved_only=True)
    print(f"[q] QS resolved depths={len(qs.lin)} QP={len(qp.lin)}", flush=True)

    df = pd.read_csv(args.traces)
    df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])
    qs5, qs08, qp5, qp08 = [], [], [], []
    for lon, lat in zip(df["path_mid_lon"].to_numpy(float), df["path_mid_lat"].to_numpy(float)):
        qs5.append(qs.sample(lon, lat, 5.0))
        qs08.append(qs.column_mean(lon, lat, 0.0, 8.0))
        qp5.append(qp.sample(lon, lat, 5.0))
        qp08.append(qp.column_mean(lon, lat, 0.0, 8.0))
    df["qs_5km"] = qs5
    df["qs_0to8"] = qs08
    df["qp_5km"] = qp5
    df["qp_0to8"] = qp08
    work = df.dropna(subset=["beta_resid", "qs_0to8"]).copy()
    print(f"[q] traces with QS={len(work)} / {len(df)}", flush=True)

    x = work["qs_0to8"].to_numpy(float)
    y = work["beta_resid"].to_numpy(float)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    work["beta_after_qs"] = y - A @ coef
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 5 else float("nan")

    rows = []
    for rank, (clon, clat, _, name) in CELLS.items():
        m = np.isclose(work["gx"], clon) & np.isclose(work["gy"], clat)
        sub, ctrl = work[m], work[~m]
        rows.append(
            {
                "rank": rank,
                "name": name,
                "n_cell": int(len(sub)),
                "qs_0to8_cell": float(sub["qs_0to8"].median()) if len(sub) else None,
                "qs_0to8_ctrl": float(ctrl["qs_0to8"].median()),
                "qs_5km_cell": float(sub["qs_5km"].median()) if len(sub) else None,
                "qp_0to8_cell": float(sub["qp_0to8"].median()) if len(sub) else None,
                "delta_qs": float(sub["qs_0to8"].median() - ctrl["qs_0to8"].median()) if len(sub) else None,
                "delta_beta": float(sub["beta_resid"].median() - ctrl["beta_resid"].median()) if len(sub) else None,
                "delta_beta_after_qs": float(sub["beta_after_qs"].median() - ctrl["beta_after_qs"].median())
                if len(sub)
                else None,
                "p10_qs_cell": float(sub["qs_0to8"].quantile(0.1)) if len(sub) else None,
                "p90_qs_cell": float(sub["qs_0to8"].quantile(0.9)) if len(sub) else None,
            }
        )
    summ = pd.DataFrame(rows)

    m2 = np.isclose(work["gx"], CELLS[2][0]) & np.isclose(work["gy"], CELLS[2][1])
    m5 = np.isclose(work["gx"], CELLS[5][0]) & np.isclose(work["gy"], CELLS[5][1])
    v2 = work.loc[m2, "qs_0to8"].to_numpy(float)
    v5 = work.loc[m5, "qs_0to8"].to_numpy(float)
    lo = max(np.nanpercentile(v2, 25), np.nanpercentile(v5, 25)) if len(v2) and len(v5) else np.nan
    hi = min(np.nanpercentile(v2, 75), np.nanpercentile(v5, 75)) if len(v2) and len(v5) else np.nan
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        lo = max(np.nanmin(v2), np.nanmin(v5)) if len(v2) and len(v5) else np.nan
        hi = min(np.nanmax(v2), np.nanmax(v5)) if len(v2) and len(v5) else np.nan
        iqr_overlap = False
    else:
        iqr_overlap = True
    split = {}
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        in_band = (work["qs_0to8"] >= lo) & (work["qs_0to8"] <= hi)
        a = work.loc[m2 & in_band, "beta_resid"].to_numpy(float)
        b = work.loc[m5 & in_band, "beta_resid"].to_numpy(float)
        split = {
            "qs_band": [float(lo), float(hi)],
            "iqr_overlap": bool(iqr_overlap),
            "n_#2": int(np.isfinite(a).sum()),
            "n_#5": int(np.isfinite(b).sum()),
            "beta_#2": float(np.nanmedian(a)) if np.isfinite(a).any() else None,
            "beta_#5": float(np.nanmedian(b)) if np.isfinite(b).any() else None,
            "delta_#5_minus_#2": float(np.nanmedian(b) - np.nanmedian(a))
            if np.isfinite(a).any() and np.isfinite(b).any()
            else None,
        }

    zs = np.arange(0.0, 21.0, 1.0)
    profiles = {}
    for rank, (clon, clat, _, name) in CELLS.items():
        profiles[str(rank)] = {
            "name": name,
            "qs": [qs.sample(clon, clat, z) for z in zs],
            "qp": [qp.sample(clon, clat, z) for z in zs],
            "z": zs.tolist(),
        }

    d_qs = None
    if summ["qs_0to8_cell"].notna().all():
        d_qs = float(
            summ.loc[summ["rank"] == 2, "qs_0to8_cell"].iloc[0]
            - summ.loc[summ["rank"] == 5, "qs_0to8_cell"].iloc[0]
        )
    left2 = float(summ.loc[summ["rank"] == 2, "delta_beta_after_qs"].iloc[0])
    left5 = float(summ.loc[summ["rank"] == 5, "delta_beta_after_qs"].iloc[0])
    n2b, n5b = split.get("n_#2", 0), split.get("n_#5", 0)
    dsplit = abs(split.get("delta_#5_minus_#2") or 0)
    if split.get("iqr_overlap") and n2b >= 20 and n5b >= 15 and dsplit > 0.015:
        verdict = "SPLIT"
        note = (
            f"#2 and #5 overlap in QS IQR [{lo:.0f},{hi:.0f}] and still differ in β_res "
            f"by {split['delta_#5_minus_#2']:+.3f}. Facies residual is not just a Q-province label."
        )
    elif d_qs is not None and abs(d_qs) > 40 and (abs(left2) < 0.01 or abs(left5) < 0.01):
        verdict = "RETRACE"
        note = (
            f"#2 vs #5 ΔQS={d_qs:+.0f}; leftover Δβ after QS is weak. "
            "Anomalies largely retrace Lin & Jordan 2023 attenuation."
        )
    else:
        verdict = "PARTIAL"
        note = (
            f"#2 vs #5 ΔQS={d_qs:+.0f} (None if missing); leftover Δβ after QS "
            f"(#2 {left2:+.3f}, #5 {left5:+.3f}). Q explains part, not all."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work[
        ["trace_name", "site", "shape", "gx", "gy", "beta_resid", "qs_5km", "qs_0to8", "qp_5km", "qp_0to8", "beta_after_qs"]
    ].to_csv(out / "beta_vs_lin2023_q.csv", index=False)
    summ.to_csv(out / "beta_vs_lin2023_q_cells.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), facecolor=BG)
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    xy5, q5 = qs.map_at(5.0)
    sc = axes[0].scatter(xy5[:, 0], xy5[:, 1], c=q5, s=8, cmap="magma", vmin=200, vmax=700, linewidths=0)
    for rank, (clon, clat, col, _) in CELLS.items():
        axes[0].plot(clon, clat, marker="s", ms=10, mfc="none", mec=col, mew=1.6)
        axes[0].text(clon + 0.08, clat + 0.08, f"#{rank}", color=col, fontsize=8, fontweight="bold")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-121.2, -114.8)
    axes[0].set_ylim(32.4, 36.2)
    axes[0].set_title("Lin & Jordan 2023 QS at 5 km (resolved nodes)", fontsize=9, color=C_INK)
    axes[0].set_xlabel("lon", fontsize=8)
    axes[0].set_ylabel("lat", fontsize=8)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("QS", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    for rank, (clon, clat, col, name) in CELLS.items():
        axes[1].plot(profiles[str(rank)]["qs"], zs, color=col, lw=1.8, label=f"#{rank} {name.split('/')[0].strip()}")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("QS", fontsize=8)
    axes[1].set_ylabel("depth (km)", fontsize=8)
    axes[1].set_title("Vertical QS at cell centers", fontsize=9, color=C_INK)
    axes[1].legend(fontsize=7, frameon=False)

    axes[2].scatter(work["qs_0to8"], work["beta_resid"], s=5, c="#D6D3D1", alpha=0.35, linewidths=0, rasterized=True)
    for rank, (clon, clat, col, _) in CELLS.items():
        m = np.isclose(work["gx"], clon) & np.isclose(work["gy"], clat)
        axes[2].scatter(
            work.loc[m, "qs_0to8"], work.loc[m, "beta_resid"], s=12, c=col, alpha=0.75, linewidths=0, label=f"#{rank}", zorder=3
        )
    xx = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 40)
    axes[2].plot(xx, coef[0] + coef[1] * xx, color=C_INK, lw=1.1)
    axes[2].set_xlabel("path-midpoint mean QS 0–8 km", fontsize=8)
    axes[2].set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=8)
    axes[2].set_title(f"β_res vs Lin QS   r={r:.2f}", fontsize=9, color=C_INK)
    axes[2].legend(fontsize=7, frameon=False)

    xs = np.arange(len(summ))
    axes[3].bar(xs - 0.18, summ["delta_beta"], width=0.36, color="#6B7280", label="before QS")
    axes[3].bar(xs + 0.18, summ["delta_beta_after_qs"], width=0.36, color="#7C3AED", label="after QS")
    axes[3].axhline(0, color=C_MUTED, lw=0.7, ls="--")
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(
        [f"#{int(r.rank)}\n{r.name.split('/')[0].strip()[:16]}" for r in summ.itertuples()], fontsize=7
    )
    axes[3].set_ylabel(r"cell − rest median $\Delta\beta$", fontsize=8)
    axes[3].set_title(f"Survive Lin QS?  verdict={verdict}", fontsize=9, color=C_INK)
    axes[3].legend(fontsize=7, frameon=False)

    fig.suptitle(note, fontsize=8.5, color=C_INK, y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "beta_vs_lin2023_q.png", dpi=args.dpi, facecolor=BG)
    fig.savefig(out / "beta_vs_lin2023_q.pdf", dpi=args.dpi, facecolor=BG)
    plt.close(fig)

    report = {
        "n_with_qs": int(len(work)),
        "pearson_beta_vs_qs0to8": r,
        "qs_coef": [float(coef[0]), float(coef[1])],
        "cells": summ.to_dict(orient="records"),
        "qs_overlap_split": split,
        "profiles": profiles,
        "verdict": verdict,
        "verdict_note": note,
        "citation": "Lin & Jordan 2023 EPSL doi:10.1016/j.epsl.2023.118227 / github.com/yupinlin/SouthernCaliforniaQmodel",
    }
    (out / "beta_vs_lin2023_q_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    slim = {k: report[k] for k in ("n_with_qs", "pearson_beta_vs_qs0to8", "cells", "qs_overlap_split", "verdict", "verdict_note")}
    print(json.dumps(slim, indent=2), flush=True)
    print(f"[done] {out / 'beta_vs_lin2023_q.png'}", flush=True)


if __name__ == "__main__":
    main()
