#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay Berg et al. 2021 SoCal upper-crust Vs / VpVs on facies residuals.

Gate A test: do cells #2 (Salton, faster decay) and #5 (ETR, slower decay)
retrace published Vs provinces, or split a Vs isosurface?

Example
-------
  PYTHONPATH=. python tools/overlay_berg2021_vs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import netcdf_file

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
    p.add_argument("--nc", default="data/external_tomo/SoCal-BergEtAl2021-UpperCrustVsandVpVs.r0.0.nc")
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _cell_key(lon, lat, grid=GRID):
    gx = np.floor(lon / grid) * grid + 0.5 * grid
    gy = np.floor(lat / grid) * grid + 0.5 * grid
    return round(float(gx), 6), round(float(gy), 6)


def load_berg(path: Path):
    ds = netcdf_file(str(path), "r", mmap=False)
    lat = np.array(ds.variables["latitude"][:], dtype=float)
    lon = np.array(ds.variables["longitude"][:], dtype=float)
    dep = np.array(ds.variables["depth"][:], dtype=float)
    vs = np.array(ds.variables["vs"][:], dtype=float)  # depth, lat, lon
    vpvs = np.array(ds.variables["vpvs"][:], dtype=float)
    ds.close()
    vs[~np.isfinite(vs)] = np.nan
    vpvs[~np.isfinite(vpvs)] = np.nan
    # RegularGridInterpolator wants ascending axes; scipy netcdf usually is.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        vs = vs[:, ::-1, :]
        vpvs = vpvs[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        vs = vs[:, :, ::-1]
        vpvs = vpvs[:, :, ::-1]
    return lat, lon, dep, vs, vpvs


def make_interp(lat, lon, dep, cube, fill=np.nan):
    # (depth, lat, lon) -> sample at (d, lat, lon)
    return RegularGridInterpolator(
        (dep, lat, lon),
        cube,
        bounds_error=False,
        fill_value=fill,
    )


def column_mean(interp, lon, lat, z_lo, z_hi, n=17):
    zs = np.linspace(z_lo, z_hi, n)
    pts = np.column_stack([zs, np.full(n, lat), np.full(n, lon)])
    vals = interp(pts)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def main() -> None:
    args = parse_args()
    lat, lon, dep, vs, vpvs = load_berg(Path(args.nc))
    f_vs = make_interp(lat, lon, dep, vs)
    f_vpvs = make_interp(lat, lon, dep, vpvs)
    print(
        f"[berg] lat {lat.min():.2f}:{lat.max():.2f} lon {lon.min():.2f}:{lon.max():.2f} "
        f"z {dep.min():.1f}:{dep.max():.1f} km  vs finite={np.isfinite(vs).mean():.2%}",
        flush=True,
    )

    df = pd.read_csv(args.traces)
    df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])

    mids = df[["path_mid_lon", "path_mid_lat"]].to_numpy(float)
    vs1, vs5, vs8, vs08, vp0 = [], [], [], [], []
    for x, y in mids:
        vs1.append(float(f_vs([[1.0, y, x]])[0]))
        vs5.append(float(f_vs([[5.0, y, x]])[0]))
        vs8.append(float(f_vs([[8.0, y, x]])[0]))
        vs08.append(column_mean(f_vs, x, y, 0.0, 8.0))
        vp0.append(float(f_vpvs([[0.5, y, x]])[0]))
    df["vs_1km"] = vs1
    df["vs_5km"] = vs5
    df["vs_8km"] = vs8
    df["vs_0to8"] = vs08
    df["vpvs_0p5"] = vp0
    work = df.dropna(subset=["beta_resid", "vs_0to8"]).copy()
    print(f"[berg] traces with Vs={len(work)} / {len(df)}", flush=True)

    # residualize beta on mean upper-crust Vs
    x = work["vs_0to8"].to_numpy(float)
    y = work["beta_resid"].to_numpy(float)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    work["beta_after_vs"] = y - A @ coef
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
                "n_ctrl": int(len(ctrl)),
                "vs_0to8_cell": float(sub["vs_0to8"].median()),
                "vs_0to8_ctrl": float(ctrl["vs_0to8"].median()),
                "vs_5km_cell": float(sub["vs_5km"].median()),
                "vs_5km_ctrl": float(ctrl["vs_5km"].median()),
                "vpvs_cell": float(sub["vpvs_0p5"].median()),
                "vpvs_ctrl": float(ctrl["vpvs_0p5"].median()),
                "delta_vs_0to8": float(sub["vs_0to8"].median() - ctrl["vs_0to8"].median()),
                "delta_beta": float(sub["beta_resid"].median() - ctrl["beta_resid"].median()),
                "delta_beta_after_vs": float(
                    sub["beta_after_vs"].median() - ctrl["beta_after_vs"].median()
                ),
                "p10_vs_cell": float(sub["vs_0to8"].quantile(0.1)),
                "p90_vs_cell": float(sub["vs_0to8"].quantile(0.9)),
            }
        )
    summ = pd.DataFrame(rows)

    # overlap / split test: common Vs band between #2 and #5
    m2 = np.isclose(work["gx"], CELLS[2][0]) & np.isclose(work["gy"], CELLS[2][1])
    m5 = np.isclose(work["gx"], CELLS[5][0]) & np.isclose(work["gy"], CELLS[5][1])
    v2 = work.loc[m2, "vs_0to8"].to_numpy(float)
    v5 = work.loc[m5, "vs_0to8"].to_numpy(float)
    lo = max(np.nanpercentile(v2, 25), np.nanpercentile(v5, 25)) if len(v2) and len(v5) else np.nan
    hi = min(np.nanpercentile(v2, 75), np.nanpercentile(v5, 75)) if len(v2) and len(v5) else np.nan
    # if IQR no overlap, use global overlapping min/max
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        lo = max(np.nanmin(v2), np.nanmin(v5)) if len(v2) and len(v5) else np.nan
        hi = min(np.nanmax(v2), np.nanmax(v5)) if len(v2) and len(v5) else np.nan
    overlap_ok = bool(np.isfinite(lo) and np.isfinite(hi) and hi - lo > 0.02)
    split = {}
    if overlap_ok:
        in_band = (work["vs_0to8"] >= lo) & (work["vs_0to8"] <= hi)
        a = work.loc[m2 & in_band, "beta_resid"].to_numpy(float)
        b = work.loc[m5 & in_band, "beta_resid"].to_numpy(float)
        c = work.loc[~m2 & ~m5 & in_band, "beta_resid"].to_numpy(float)
        split = {
            "vs_band": [float(lo), float(hi)],
            "n_#2": int(np.isfinite(a).sum()),
            "n_#5": int(np.isfinite(b).sum()),
            "n_ctrl": int(np.isfinite(c).sum()),
            "beta_#2": float(np.nanmedian(a)) if len(a) else None,
            "beta_#5": float(np.nanmedian(b)) if len(b) else None,
            "beta_ctrl": float(np.nanmedian(c)) if len(c) else None,
            "delta_#5_minus_#2": float(np.nanmedian(b) - np.nanmedian(a)) if len(a) and len(b) else None,
        }

    # depth profiles at cell centers + regional median where finite
    zs = np.linspace(0.0, 20.0, 41)
    profiles = {}
    for rank, (clon, clat, _, name) in CELLS.items():
        profiles[str(rank)] = {
            "name": name,
            "lon": clon,
            "lat": clat,
            "z": zs.tolist(),
            "vs": [float(f_vs([[z, clat, clon]])[0]) for z in zs],
            "vpvs": [float(f_vpvs([[z, clat, clon]])[0]) for z in zs],
        }
    # regional median profile on grid nodes inside SoCal data bbox of traces
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    reg_vs = []
    for z in zs:
        k = int(np.argmin(np.abs(dep - z)))
        sl = vs[k]
        reg_vs.append(float(np.nanmedian(sl)))
    profiles["regional_median"] = {"z": zs.tolist(), "vs": reg_vs}

    # verdict
    d_vs = float(summ.loc[summ["rank"] == 2, "vs_0to8_cell"].iloc[0] - summ.loc[summ["rank"] == 5, "vs_0to8_cell"].iloc[0])
    d_beta_left = float(summ.loc[summ["rank"] == 2, "delta_beta_after_vs"].iloc[0])
    d_beta5_left = float(summ.loc[summ["rank"] == 5, "delta_beta_after_vs"].iloc[0])
    same_sign_left = np.sign(d_beta_left) < 0 and np.sign(d_beta5_left) > 0
    if overlap_ok and split.get("n_#2", 0) >= 20 and split.get("n_#5", 0) >= 15 and abs(split.get("delta_#5_minus_#2") or 0) > 0.015:
        verdict = "SPLIT"
        verdict_note = (
            f"#2 and #5 overlap in Vs [{lo:.2f},{hi:.2f}] km/s and still differ in β_res "
            f"by {split['delta_#5_minus_#2']:+.3f}. Facies residual is not just a Vs province label."
        )
    elif abs(d_vs) > 0.15 and (not same_sign_left or abs(d_beta_left) < 0.008):
        verdict = "RETRACE"
        verdict_note = (
            f"#2 vs #5 ΔVs(0–8 km)={d_vs:+.2f} km/s; leftover Δβ after Vs is weak. "
            "Anomalies largely retrace published upper-crust Vs."
        )
    else:
        verdict = "PARTIAL"
        verdict_note = (
            f"#2 vs #5 sit in different Vs regimes (ΔVs={d_vs:+.2f} km/s) but leftover "
            f"Δβ after Vs remains (#2 {d_beta_left:+.3f}, #5 {d_beta5_left:+.3f}). "
            "Velocity structure explains part, not all."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work[
        [
            "trace_name",
            "site",
            "shape",
            "gx",
            "gy",
            "beta_resid",
            "vs_1km",
            "vs_5km",
            "vs_8km",
            "vs_0to8",
            "vpvs_0p5",
            "beta_after_vs",
        ]
    ].to_csv(out / "beta_vs_berg2021.csv", index=False)
    summ.to_csv(out / "beta_vs_berg2021_cells.csv", index=False)

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), facecolor=BG)
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    # map Vs at 5 km
    k5 = int(np.argmin(np.abs(dep - 5.0)))
    vs5map = vs[k5]
    lon2, lat2 = np.meshgrid(lon, lat)
    im = axes[0].pcolormesh(lon2, lat2, vs5map, cmap="cividis", shading="auto", vmin=1.6, vmax=3.6)
    for rank, (clon, clat, col, name) in CELLS.items():
        axes[0].plot(clon, clat, marker="s", ms=10, mfc="none", mec=col, mew=1.6)
        axes[0].text(clon + 0.06, clat + 0.06, f"#{rank}", color=col, fontsize=8, fontweight="bold")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Berg 2021 Vs at 5 km + anomaly cells", fontsize=9, color=C_INK)
    axes[0].set_xlabel("lon", fontsize=8)
    axes[0].set_ylabel("lat", fontsize=8)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("Vs (km/s)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # depth profiles
    for rank, (clon, clat, col, name) in CELLS.items():
        axes[1].plot(profiles[str(rank)]["vs"], zs, color=col, lw=1.8, label=f"#{rank} {name.split('/')[0].strip()}")
    axes[1].plot(reg_vs, zs, color=C_MUTED, lw=1.1, ls="--", label="model median")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Vs (km/s)", fontsize=8)
    axes[1].set_ylabel("depth (km)", fontsize=8)
    axes[1].set_title("Vertical Vs at cell centers", fontsize=9, color=C_INK)
    axes[1].legend(fontsize=7, frameon=False)

    axes[2].scatter(work["vs_0to8"], work["beta_resid"], s=5, c="#D6D3D1", alpha=0.35, linewidths=0, rasterized=True)
    for rank, (clon, clat, col, name) in CELLS.items():
        m = np.isclose(work["gx"], clon) & np.isclose(work["gy"], clat)
        axes[2].scatter(work.loc[m, "vs_0to8"], work.loc[m, "beta_resid"], s=12, c=col, alpha=0.7, linewidths=0, label=f"#{rank}", zorder=3)
    xx = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 40)
    axes[2].plot(xx, coef[0] + coef[1] * xx, color=C_INK, lw=1.1)
    axes[2].set_xlabel("path-midpoint mean Vs 0–8 km (km/s)", fontsize=8)
    axes[2].set_ylabel(r"$\beta_{\mathrm{res}}$", fontsize=8)
    axes[2].set_title(f"β_res vs Berg Vs   r={r:.2f}", fontsize=9, color=C_INK)
    axes[2].legend(fontsize=7, frameon=False)

    xs = np.arange(len(summ))
    axes[3].bar(xs - 0.18, summ["delta_beta"], width=0.36, color="#6B7280", label="before Vs")
    axes[3].bar(xs + 0.18, summ["delta_beta_after_vs"], width=0.36, color="#B45309", label="after Vs")
    axes[3].axhline(0, color=C_MUTED, lw=0.7, ls="--")
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(
        [f"#{int(r.rank)}\n{r.name.split('/')[0].strip()[:16]}" for r in summ.itertuples()], fontsize=7
    )
    axes[3].set_ylabel(r"cell − rest median $\Delta\beta$", fontsize=8)
    axes[3].set_title(f"Survive Berg Vs?  verdict={verdict}", fontsize=9, color=C_INK)
    axes[3].legend(fontsize=7, frameon=False)

    fig.suptitle(verdict_note, fontsize=8.5, color=C_INK, y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "beta_vs_berg2021.png", dpi=args.dpi, facecolor=BG)
    fig.savefig(out / "beta_vs_berg2021.pdf", dpi=args.dpi, facecolor=BG)
    plt.close(fig)

    report = {
        "n_with_vs": int(len(work)),
        "pearson_beta_vs_0to8": r,
        "vs_coef": [float(coef[0]), float(coef[1])],
        "cells": summ.to_dict(orient="records"),
        "vs_overlap_split": split,
        "profiles": profiles,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "citation": "Berg et al. 2021 GRL doi:10.1029/2021GL092626 / IRIS EMC",
    }
    (out / "beta_vs_berg2021_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_with_vs", "pearson_beta_vs_0to8", "cells", "vs_overlap_split", "verdict", "verdict_note")}, indent=2), flush=True)
    print(f"[done] {out / 'beta_vs_berg2021.png'}", flush=True)


if __name__ == "__main__":
    main()
