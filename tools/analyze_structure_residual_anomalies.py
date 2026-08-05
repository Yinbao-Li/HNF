#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structure-expectation residuals of interpretable facies (SoCal first).

Pipeline
--------
1. Load expanded regional labels (fixed stations preferred).
2. Compute distance-to-nearest Quaternary fault (USGS QFaults).
3. Fit structure expectation:
      β_res ~ log(dist) + depth + log1p(fault_dist) + station FE (ridge)
      P(multipath), P(slow_coda), P(impulsive) ~ same covariates (ridge-logit)
4. Grid residual maps and rank spatially coherent anomaly cells.
5. Write Top-K anomaly dossier + publication-style figure.

Example
-------
  PYTHONPATH=. python tools/analyze_structure_residual_anomalies.py \\
    --traces outputs/shape_labels_expanded/socal/traces_labeled.csv \\
    --faults docs/figures/geo/qfaults_socal.geojson \\
    --output-dir outputs/structure_residual_socal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.analyze_shape_temporal_evolution import REGIONS, SHAPE_COLORS, SHAPE_ORDER

C_INK = "#1C1917"
C_MUTED = "#78716C"
C_LINE = "#D6D3D1"
BG = "#FFFEF9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structure residual anomalies from waveform facies")
    p.add_argument("--traces", default="outputs/shape_labels_expanded/socal/traces_labeled.csv")
    p.add_argument("--faults", default="docs/figures/geo/qfaults_socal.geojson")
    p.add_argument("--faults-offshore", default="docs/figures/geo/qfaults_socal_offshore.geojson")
    p.add_argument(
        "--extra-faults",
        default="",
        help="Comma-separated extra fault GeoJSON paths",
    )
    p.add_argument("--region", default="socal")
    p.add_argument("--fixed-only", action="store_true", default=True)
    p.add_argument("--no-fixed-only", action="store_false", dest="fixed_only")
    p.add_argument("--dist-min", type=float, default=20.0)
    p.add_argument("--dist-max", type=float, default=120.0)
    p.add_argument("--grid-deg", type=float, default=0.25)
    p.add_argument("--min-cell-n", type=int, default=25)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--ridge", type=float, default=8.0)
    p.add_argument("--coastline", default="docs/figures/geo/ne_110m_land.geojson")
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _load_polylines(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        return []
    geo = json.loads(path.read_text(encoding="utf-8"))
    lines: list[np.ndarray] = []

    def _add(coords):
        arr = np.asarray(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            lines.append(arr[:, :2])

    for feat in geo.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue
        if gtype == "LineString":
            _add(coords)
        elif gtype == "MultiLineString":
            for part in coords:
                _add(part)
        elif gtype == "Polygon":
            _add(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                _add(poly[0])
    return lines


def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _densify_line(xy: np.ndarray, step_km: float = 2.0) -> np.ndarray:
    if len(xy) < 2:
        return xy
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        d = float(_haversine_km(a[0], a[1], b[0], b[1]))
        n = max(int(np.ceil(d / step_km)), 1)
        if n <= 1:
            out.append(b)
            continue
        for i in range(1, n + 1):
            t = i / n
            out.append(a * (1 - t) + b * t)
    return np.vstack(out)


def fault_point_cloud(lines: list[np.ndarray], step_km: float = 2.5) -> np.ndarray:
    pts = []
    for ln in lines:
        dens = _densify_line(ln, step_km=step_km)
        pts.append(dens)
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)


def nearest_fault_km(lons: np.ndarray, lats: np.ndarray, fault_xy: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Approximate nearest-fault distance via local degree→km scaling (fast, SoCal-scale)."""
    if fault_xy.size == 0:
        return np.full(len(lons), np.nan)
    out = np.empty(len(lons), dtype=float)
    fx = fault_xy[:, 0].astype(np.float64)
    fy = fault_xy[:, 1].astype(np.float64)
    # subsample very dense fault clouds
    if len(fx) > 80000:
        idx = np.linspace(0, len(fx) - 1, 80000).astype(int)
        fx, fy = fx[idx], fy[idx]
    for i0 in range(0, len(lons), chunk):
        i1 = min(len(lons), i0 + chunk)
        lon = lons[i0:i1][:, None]
        lat = lats[i0:i1][:, None]
        dlat = lat - fy[None, :]
        dlon = (lon - fx[None, :]) * np.cos(np.radians(lat))
        ddeg2 = dlat * dlat + dlon * dlon
        out[i0:i1] = 111.32 * np.sqrt(ddeg2.min(axis=1))
    return out


def _design_matrix(df: pd.DataFrame, station_col: str = "site") -> tuple[np.ndarray, list[str], np.ndarray]:
    logd = np.log10(df["dist_km"].to_numpy(float) + 1.0)
    depth = np.clip(df["depth_km"].to_numpy(float), 0.0, 80.0)
    depth = np.nan_to_num(depth, nan=float(np.nanmedian(depth[np.isfinite(depth)])))
    fdist = np.log1p(np.clip(df["fault_dist_km"].to_numpy(float), 0.0, None))
    mag = df["mag"].to_numpy(float)
    mag = np.nan_to_num(mag, nan=float(np.nanmedian(mag[np.isfinite(mag)])) if np.isfinite(mag).any() else 0.0)
    X_base = np.column_stack([np.ones(len(df)), logd, depth, fdist, mag])
    names = ["intercept", "log10_dist", "depth_km", "log1p_fault_km", "mag"]
    sites = df[station_col].astype(str).to_numpy()
    uniq, inv = np.unique(sites, return_inverse=True)
    # drop last dummy to avoid collinearity with intercept
    if len(uniq) > 1:
        dummy = np.eye(len(uniq), dtype=float)[inv][:, :-1]
        X = np.column_stack([X_base, dummy])
        names += [f"site::{s}" for s in uniq[:-1]]
    else:
        X = X_base
    return X, names, inv


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xtx = X.T @ X
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, X.T @ y)


def ridge_logit(X: np.ndarray, y: np.ndarray, lam: float, n_iter: int = 25) -> np.ndarray:
    """IRLS logistic ridge."""
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = np.clip(X @ w, -20, 20)
        p = 1.0 / (1.0 + np.exp(-eta))
        s = np.clip(p * (1 - p), 1e-6, None)
        z = eta + (y - p) / s
        sw = np.sqrt(s)
        Xs = X * sw[:, None]
        zs = z * sw
        xtx = Xs.T @ Xs
        reg = lam * np.eye(X.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(xtx + reg, Xs.T @ zs)
    return w


def _r2(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def grid_stats(df: pd.DataFrame, value_col: str, grid: float, min_n: int) -> pd.DataFrame:
    lon = 0.5 * (df["src_lon"] + df["rcv_lon"])
    lat = 0.5 * (df["src_lat"] + df["rcv_lat"])
    gx = np.floor(lon / grid) * grid + 0.5 * grid
    gy = np.floor(lat / grid) * grid + 0.5 * grid
    tmp = df.copy()
    tmp["_gx"] = gx
    tmp["_gy"] = gy
    rows = []
    for (x, y), sub in tmp.groupby(["_gx", "_gy"]):
        if len(sub) < min_n:
            continue
        v = sub[value_col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size < min_n:
            continue
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        rows.append(
            {
                "lon": float(x),
                "lat": float(y),
                "n": int(len(sub)),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "se": se,
                "z": float(v.mean() / se) if np.isfinite(se) and se > 1e-12 else np.nan,
                "frac_multipath": float((sub["shape"] == "multipath").mean()),
                "frac_slow": float((sub["shape"] == "slow_coda").mean()),
                "frac_impulsive": float((sub["shape"] == "impulsive_fastQ").mean()),
                "fault_dist_med": float(sub["fault_dist_km"].median()),
                "dist_med": float(sub["dist_km"].median()),
                "depth_med": float(sub["depth_km"].median()) if sub["depth_km"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _load_coast(path: Path):
    if not path.is_file():
        return []
    geo = json.loads(path.read_text(encoding="utf-8"))
    rings = []
    for feat in geo.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Polygon":
            rings.append(np.asarray(geom["coordinates"][0]))
        elif geom.get("type") == "MultiPolygon":
            for poly in geom["coordinates"]:
                rings.append(np.asarray(poly[0]))
    return rings


def plot_maps(df, cells_beta, cells_mp, cells_sc, faults, coast, region_meta, out_png, dpi, top_cells):
    lat_lim, lon_lim = region_meta["lat"], region_meta["lon"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), facecolor=BG, sharex=True, sharey=True)
    axes = axes.ravel()
    titles = [
        r"Observed median $\beta_{\mathrm{res}}$",
        r"Structure residual of $\beta_{\mathrm{res}}$",
        "Residual P(multipath)  (obs − expected)",
        "Residual P(slow_coda)  (obs − expected)",
    ]
    datasets = [
        ("beta", df, "coda_path_residual", "coolwarm", None),
    ]

    def _base(ax):
        ax.set_facecolor(BG)
        for ring in coast:
            ax.plot(ring[:, 0], ring[:, 1], color="#B8B2A8", lw=0.4, zorder=0)
        if faults:
            segs = [ln for ln in faults if ln.shape[0] >= 2]
            lc = LineCollection(segs, colors="#4B5563", linewidths=0.45, alpha=0.75, zorder=1)
            ax.add_collection(lc)
        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color=C_LINE, lw=0.4)
        ax.tick_params(labelsize=7)

    # observed beta scatter density via hex? use cell means if available else scatter
    _base(axes[0])
    sc = axes[0].scatter(
        0.5 * (df["src_lon"] + df["rcv_lon"]),
        0.5 * (df["src_lat"] + df["rcv_lat"]),
        c=df["coda_path_residual"],
        s=6,
        cmap="coolwarm",
        vmin=-0.08,
        vmax=0.08,
        alpha=0.55,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )
    fig.colorbar(sc, ax=axes[0], shrink=0.72, pad=0.01).set_label(r"$\beta_{\mathrm{res}}$", fontsize=8)

    def _cell_map(ax, cells, col, cmap, vabs):
        _base(ax)
        if cells.empty:
            return
        sc2 = ax.scatter(
            cells["lon"],
            cells["lat"],
            c=cells[col],
            s=np.clip(cells["n"] * 1.8, 20, 140),
            cmap=cmap,
            vmin=-vabs,
            vmax=vabs,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
        fig.colorbar(sc2, ax=ax, shrink=0.72, pad=0.01)

    _cell_map(axes[1], cells_beta, "mean", "coolwarm", 0.035)
    _cell_map(axes[2], cells_mp, "mean", "PiYG", 0.18)
    _cell_map(axes[3], cells_sc, "mean", "PuOr", 0.12)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9, color=C_INK, pad=4)

    # mark top anomalies on residual-beta panel
    for i, row in enumerate(top_cells.itertuples(), start=1):
        axes[1].annotate(
            str(i),
            xy=(row.lon, row.lat),
            fontsize=8,
            fontweight="bold",
            color=C_INK,
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=C_INK, lw=0.6),
        )

    axes[0].set_ylabel("Latitude (°)")
    axes[2].set_ylabel("Latitude (°)")
    axes[2].set_xlabel("Longitude (°)")
    axes[3].set_xlabel("Longitude (°)")
    fig.suptitle(
        f"{region_meta['title']}: interpretable facies vs structure expectation\n"
        "Residual = observed − model(log dist, depth, mag, log fault-distance, station FE)",
        fontsize=11,
        color=C_INK,
        y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=BG)
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, facecolor=BG)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    region = args.region
    if region not in REGIONS:
        raise SystemExit(f"unknown region {region}")
    meta = REGIONS[region]

    df = pd.read_csv(args.traces)
    if "fixed_station" in df.columns and args.fixed_only:
        df = df[df["fixed_station"].astype(str).str.lower().isin(["1", "true", "yes"])].copy()
    df = df[df["shape"].isin(SHAPE_ORDER)].copy()
    df["dist_km"] = pd.to_numeric(df["dist_km"], errors="coerce")
    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df["coda_path_residual"] = pd.to_numeric(df["coda_path_residual"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df = df[df["dist_km"].between(args.dist_min, args.dist_max)]
    df = df.dropna(subset=["src_lat", "src_lon", "rcv_lat", "rcv_lon", "coda_path_residual", "dist_km"])
    if "site" not in df.columns:
        df["site"] = df["network"].astype(str) + "." + df["station"].astype(str)
    print(f"[in] traces after filters: {len(df)}  sites={df['site'].nunique()}", flush=True)

    lines = _load_polylines(Path(args.faults)) + _load_polylines(Path(args.faults_offshore))
    if args.extra_faults.strip():
        for pth in args.extra_faults.split(","):
            pth = pth.strip()
            if pth:
                lines += _load_polylines(Path(pth))
    print(f"[faults] polylines={len(lines)}", flush=True)
    fxy = fault_point_cloud(lines, step_km=4.0)
    print(f"[faults] sample points={len(fxy)}", flush=True)

    mid_lon = 0.5 * (df["src_lon"].to_numpy(float) + df["rcv_lon"].to_numpy(float))
    mid_lat = 0.5 * (df["src_lat"].to_numpy(float) + df["rcv_lat"].to_numpy(float))
    df["path_mid_lon"] = mid_lon
    df["path_mid_lat"] = mid_lat
    df["fault_dist_km"] = nearest_fault_km(mid_lon, mid_lat, fxy)
    print(
        f"[faults] dist km percentiles: "
        + ", ".join(f"p{q}={np.nanpercentile(df['fault_dist_km'], q):.1f}" for q in (10, 50, 90)),
        flush=True,
    )

    X, names, _ = _design_matrix(df)
    yb = df["coda_path_residual"].to_numpy(float)
    wb = ridge_fit(X, yb, args.ridge)
    df["beta_expected"] = X @ wb
    df["beta_resid"] = yb - df["beta_expected"]
    beta_r2 = _r2(yb, df["beta_expected"])

    shape_models = {}
    for sh, col in (
        ("multipath", "mp"),
        ("slow_coda", "sc"),
        ("impulsive_fastQ", "imp"),
    ):
        y = (df["shape"] == sh).astype(float).to_numpy()
        w = ridge_logit(X, y, args.ridge)
        eta = np.clip(X @ w, -20, 20)
        p = 1.0 / (1.0 + np.exp(-eta))
        df[f"p_{col}_exp"] = p
        df[f"p_{col}_obs"] = y
        df[f"p_{col}_resid"] = y - p
        shape_models[sh] = {
            "prevalence": float(y.mean()),
            "coef_base": {n: float(w[i]) for i, n in enumerate(names) if not n.startswith("site::")},
        }

    coef_beta = {n: float(wb[i]) for i, n in enumerate(names) if not n.startswith("site::")}
    print(f"[fit] beta R²={beta_r2:.3f}  base coefs={coef_beta}", flush=True)

    cells_beta = grid_stats(df.assign(coda_path_residual=df["beta_resid"]), "coda_path_residual", args.grid_deg, args.min_cell_n)
    cells_mp = grid_stats(df.assign(coda_path_residual=df["p_mp_resid"]), "coda_path_residual", args.grid_deg, args.min_cell_n)
    cells_sc = grid_stats(df.assign(coda_path_residual=df["p_sc_resid"]), "coda_path_residual", args.grid_deg, args.min_cell_n)
    cells_imp = grid_stats(df.assign(coda_path_residual=df["p_imp_resid"]), "coda_path_residual", args.grid_deg, args.min_cell_n)

    # anomaly score: combine |z| of beta residual and multipath residual
    cells = cells_beta.rename(columns={"mean": "beta_resid_mean", "z": "beta_z", "se": "beta_se"}).copy()
    if not cells_mp.empty:
        cells = cells.merge(
            cells_mp[["lon", "lat", "mean", "z"]].rename(columns={"mean": "mp_resid_mean", "z": "mp_z"}),
            on=["lon", "lat"],
            how="left",
        )
    if not cells_sc.empty:
        cells = cells.merge(
            cells_sc[["lon", "lat", "mean", "z"]].rename(columns={"mean": "sc_resid_mean", "z": "sc_z"}),
            on=["lon", "lat"],
            how="left",
        )
    cells["score"] = np.sqrt(
        np.nan_to_num(cells.get("beta_z", np.nan) ** 2, nan=0.0)
        + np.nan_to_num(cells.get("mp_z", np.nan) ** 2, nan=0.0)
        + 0.6 * np.nan_to_num(cells.get("sc_z", np.nan) ** 2, nan=0.0)
    )
    # require some evidence in z
    cells = cells.replace([np.inf, -np.inf], np.nan)
    top = cells.sort_values("score", ascending=False).head(args.top_k).reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_out_cols = [
        "trace_name", "shape", "mag", "dist_km", "depth_km", "site", "fixed_station",
        "src_lat", "src_lon", "rcv_lat", "rcv_lon", "path_mid_lat", "path_mid_lon",
        "fault_dist_km", "coda_path_residual", "beta_expected", "beta_resid",
        "p_mp_exp", "p_mp_resid", "p_sc_exp", "p_sc_resid", "p_imp_exp", "p_imp_resid",
    ]
    keep = [c for c in df_out_cols if c in df.columns]
    df[keep].to_csv(out / "traces_with_structure.csv", index=False)
    cells.to_csv(out / "grid_cells.csv", index=False)
    top.to_csv(out / "top_anomalies.csv", index=False)
    cells_beta.to_csv(out / "grid_beta_resid.csv", index=False)
    cells_mp.to_csv(out / "grid_multipath_resid.csv", index=False)
    cells_sc.to_csv(out / "grid_slowcoda_resid.csv", index=False)

    coast = _load_coast(Path(args.coastline))
    plot_maps(
        df,
        cells_beta,
        cells_mp,
        cells_sc,
        lines,
        coast,
        meta,
        out / f"{region}_structure_residuals.png",
        args.dpi,
        top,
    )

    # dossier text
    lines_md = [
        f"# Structure residual anomalies — {meta['title']}",
        "",
        f"- Traces used: **{len(df)}** (dist {args.dist_min:.0f}–{args.dist_max:.0f} km"
        + (", fixed stations only" if args.fixed_only else "")
        + ")",
        f"- Stations: **{df['site'].nunique()}**; fault polylines: **{len(lines)}**",
        f"- Structure model R² for β_res: **{beta_r2:.3f}**",
        f"- Base coefficients (β_res): `{json.dumps({k: round(v, 4) for k, v in coef_beta.items()})}`",
        "",
        "## How to read residuals",
        "- Residual > 0 in β_res: coda decays **slower than expected** given distance/depth/fault proximity/station.",
        "- Residual > 0 in P(multipath): more multi-lobe ρ(τ) than structure expectation.",
        "- New-structure candidates are **spatially coherent cells with large |z| after structure regression**.",
        "",
        "## Top anomaly cells",
        "",
        "| rank | lon | lat | n | β_res resid | β z | mp resid | mp z | sc resid | score | fault dist (km) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top.itertuples():
        lines_md.append(
            f"| {r.rank} | {r.lon:.2f} | {r.lat:.2f} | {r.n} | "
            f"{getattr(r, 'beta_resid_mean', float('nan')):+.4f} | {getattr(r, 'beta_z', float('nan')):+.2f} | "
            f"{getattr(r, 'mp_resid_mean', float('nan')):+.3f} | {getattr(r, 'mp_z', float('nan')):+.2f} | "
            f"{getattr(r, 'sc_resid_mean', float('nan')):+.3f} | {r.score:.2f} | {r.fault_dist_med:.1f} |"
        )
    lines_md += [
        "",
        "## Next validation (to convert candidates → publication claim)",
        "1. Inspect each Top cell against geologic maps (basin edges, step-overs, volcanic fields).",
        "2. Same-station split: events whose path midpoint falls in the cell vs nearby control cells.",
        "3. Compare with independent Qc / velocity tomography at the same cells.",
        "4. Replicate the strongest cell family in PNW or Alaska with the same pipeline.",
        "",
        f"Figure: `{region}_structure_residuals.png`",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines_md), encoding="utf-8")

    summary = {
        "region": region,
        "n_traces": int(len(df)),
        "n_sites": int(df["site"].nunique()),
        "n_fault_lines": int(len(lines)),
        "beta_r2": round(beta_r2, 4),
        "beta_base_coef": coef_beta,
        "shape_models": shape_models,
        "n_grid_cells": int(len(cells)),
        "top_anomalies": top.to_dict(orient="records"),
        "filters": {
            "fixed_only": bool(args.fixed_only),
            "dist_min": args.dist_min,
            "dist_max": args.dist_max,
            "grid_deg": args.grid_deg,
            "min_cell_n": args.min_cell_n,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[done] {out / 'REPORT.md'}", flush=True)
    print(top[["rank", "lon", "lat", "n", "beta_resid_mean", "beta_z", "score"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
