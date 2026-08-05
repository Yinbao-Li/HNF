#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-station validation + geologic tagging of structure-residual cells.

For each Top anomaly cell, compare traces whose path midpoint falls in the cell
against control traces recorded at the *same stations* whose midpoints fall
outside the cell (and outside a small buffer). This tests whether the anomaly
is a path-structure effect rather than a station-mix artifact.

Example
-------
  PYTHONPATH=. python tools/validate_structure_anomaly_cells.py \\
    --traces outputs/structure_residual_socal/traces_with_structure.csv \\
    --top outputs/structure_residual_socal/top_anomalies.csv \\
    --output-dir outputs/structure_residual_socal
"""

from __future__ import annotations

import argparse
import json
from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_INK = "#1C1917"
C_MUTED = "#78716C"
C_LINE = "#D6D3D1"
BG = "#FFFEF9"

# Hand-curated geologic tags for 0.25° SoCal cells (publication working names).
GEO_TAGS = {
    (-116.875, 33.375): "San Jacinto FZ / Anza–Coyote Creek step",
    (-116.375, 33.375): "Salton Trough / S. San Andreas–Imperial junction",
    (-117.125, 33.625): "Elsinore–Temecula / Peninsular Ranges transition",
    (-116.375, 32.625): "Cerro Prieto–Imperial Valley south",
    (-116.375, 33.875): "E. Transverse Ranges / Little San Bernardino Mts",
    (-116.875, 33.875): "San Andreas–San Jacinto junction (San Gorgonio–Banning)",
    (-116.625, 33.625): "San Jacinto FZ central (Anza–Hemet)",
    (-120.125, 35.875): "Central Coast / Hosgri–Rinconada corridor",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate structure anomaly cells")
    p.add_argument("--traces", default="outputs/structure_residual_socal/traces_with_structure.csv")
    p.add_argument("--top", default="outputs/structure_residual_socal/top_anomalies.csv")
    p.add_argument("--grid-deg", type=float, default=0.25)
    p.add_argument("--buffer-cells", type=float, default=0.5, help="exclude this many cells around target as neither in/out")
    p.add_argument("--min-pair-n", type=int, default=8)
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--geo-tag", default="", help="optional JSON dict {(lon,lat): name} overrides")
    return p.parse_args()


def _mw_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().to_numpy()
    ra = ranks[: len(a)].sum()
    na, nb = len(a), len(b)
    u = ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    sig = np.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (u - mu) / max(sig, 1e-9)
    p = float(erfc(abs(z) / sqrt(2.0)))
    return float(z), p


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def _cell_key(lon: float, lat: float, grid: float) -> tuple[float, float]:
    gx = np.floor(lon / grid) * grid + 0.5 * grid
    gy = np.floor(lat / grid) * grid + 0.5 * grid
    return round(float(gx), 6), round(float(gy), 6)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.traces)
    top = pd.read_csv(args.top)
    grid = args.grid_deg
    geo_tags = dict(GEO_TAGS)
    if args.geo_tag.strip():
        extra = json.loads(args.geo_tag)
        for k, v in extra.items():
            lon, lat = [float(x) for x in k.split(",")]
            geo_tags[(round(lon, 3), round(lat, 3))] = str(v)
    df["gx"], df["gy"] = zip(*[_cell_key(lo, la, grid) for lo, la in zip(df["path_mid_lon"], df["path_mid_lat"])])

    rows = []
    station_rows = []
    for rec in top.itertuples(index=False):
        clon, clat = float(rec.lon), float(rec.lat)
        tag = geo_tags.get((round(clon, 3), round(clat, 3))) or geo_tags.get((clon, clat))
        if tag is None:
            # fuzzy match
            for (glon, glat), name in geo_tags.items():
                if abs(glon - clon) < 1e-3 and abs(glat - clat) < 1e-3:
                    tag = name
                    break
        tag = tag or "unlabeled cell"

        in_m = (np.isclose(df["gx"], clon)) & (np.isclose(df["gy"], clat))
        buf = args.buffer_cells * grid
        near = (np.abs(df["gx"] - clon) <= buf + 1e-9) & (np.abs(df["gy"] - clat) <= buf + 1e-9)
        ctrl_pool = df[~near]
        inside = df[in_m]
        sites = sorted(set(inside["site"].astype(str)) & set(ctrl_pool["site"].astype(str)))
        paired_in, paired_out = [], []
        site_stats = []
        for site in sites:
            a = inside.loc[inside["site"] == site, "beta_resid"].to_numpy(float)
            b = ctrl_pool.loc[ctrl_pool["site"] == site, "beta_resid"].to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                continue
            paired_in.append(a)
            paired_out.append(b)
            site_stats.append(
                {
                    "rank": int(rec.rank),
                    "site": site,
                    "n_in": int(len(a)),
                    "n_out": int(len(b)),
                    "beta_in": float(np.nanmedian(a)),
                    "beta_out": float(np.nanmedian(b)),
                    "delta": float(np.nanmedian(a) - np.nanmedian(b)),
                    "frac_mp_in": float((inside.loc[inside["site"] == site, "shape"] == "multipath").mean()) if "shape" in inside.columns else np.nan,
                    "frac_mp_out": float((ctrl_pool.loc[ctrl_pool["site"] == site, "shape"] == "multipath").mean()) if "shape" in ctrl_pool.columns else np.nan,
                }
            )
        if not paired_in:
            rows.append(
                {
                    "rank": int(rec.rank),
                    "lon": clon,
                    "lat": clat,
                    "geo_tag": tag,
                    "n_cell": int(len(inside)),
                    "n_sites_paired": 0,
                    "status": "insufficient same-station pairs",
                }
            )
            continue
        a_all = np.concatenate(paired_in)
        b_all = np.concatenate(paired_out)
        z, p = _mw_p(a_all, b_all)
        d = _cohen_d(a_all, b_all)
        site_delta = np.array([s["delta"] for s in site_stats], float)
        rows.append(
            {
                "rank": int(rec.rank),
                "lon": clon,
                "lat": clat,
                "geo_tag": tag,
                "n_cell": int(len(inside)),
                "n_sites_paired": int(len(site_stats)),
                "n_in_paired": int(len(a_all)),
                "n_out_paired": int(len(b_all)),
                "beta_in_med": float(np.nanmedian(a_all)),
                "beta_out_med": float(np.nanmedian(b_all)),
                "delta_med": float(np.nanmedian(a_all) - np.nanmedian(b_all)),
                "site_delta_med": float(np.nanmedian(site_delta)),
                "site_delta_frac_neg": float((site_delta < 0).mean()),
                "cohen_d": d,
                "mw_z": z,
                "mw_p": p,
                "status": "ok" if len(site_stats) >= args.min_pair_n else "few stations",
            }
        )
        station_rows.extend(site_stats)

    res = pd.DataFrame(rows).sort_values("rank")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "same_station_validation.csv", index=False)
    if station_rows:
        pd.DataFrame(station_rows).to_csv(out / "same_station_by_site.csv", index=False)

    # figure: per-cell same-station delta
    ok = res[res["n_sites_paired"] > 0].copy()
    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhline(0.0, color=C_MUTED, lw=0.8, ls="--")
    xs = np.arange(len(ok))
    cols = ["#B91C1C" if v < 0 else "#1D4ED8" for v in ok["site_delta_med"]]
    ax.bar(xs, ok["site_delta_med"], color=cols, alpha=0.85, width=0.72)
    ax.set_xticks(xs)
    labels = [f"#{int(r.rank)}\n{r.geo_tag.split('/')[0].strip()[:18]}" for r in ok.itertuples()]
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"Same-station median $\Delta\beta_{\mathrm{res}}$ (cell − control)", color=C_INK)
    ax.set_title(
        "Path-midpoint cell vs same-station controls\nnegative = faster coda decay than the same stations see elsewhere",
        color=C_INK,
        fontsize=10,
    )
    for i, r in enumerate(ok.itertuples()):
        ax.text(i, r.site_delta_med + (0.0015 if r.site_delta_med >= 0 else -0.0015), f"n_st={int(r.n_sites_paired)}\np≈{r.mw_p:.1g}", ha="center", va="bottom" if r.site_delta_med >= 0 else "top", fontsize=6.2, color=C_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "same_station_validation.png", dpi=args.dpi, facecolor=BG)
    fig.savefig(out / "same_station_validation.pdf", dpi=args.dpi, facecolor=BG)
    plt.close(fig)

    # update dossier
    lines = [
        "# Same-station validation of Top structure-residual cells",
        "",
        "Control = traces at the **same station**, path midpoint outside the cell and a 0.5-cell buffer.",
        "If the anomaly survives this split, it is unlikely to be only a station-composition artifact.",
        "",
        "| rank | geologic working name | sites | Δβ site-med | % sites Δβ<0 | Cohen d | MW p | status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in res.itertuples():
        if int(r.n_sites_paired) == 0:
            lines.append(f"| {int(r.rank)} | {r.geo_tag} | 0 |  |  |  |  | {r.status} |")
            continue
        lines.append(
            f"| {int(r.rank)} | {r.geo_tag} | {int(r.n_sites_paired)} | "
            f"{r.site_delta_med:+.4f} | {100*r.site_delta_frac_neg:.0f}% | {r.cohen_d:+.2f} | {r.mw_p:.2g} | {r.status} |"
        )

    survivors = res[(res["n_sites_paired"] >= args.min_pair_n) & (res["mw_p"] < 0.05)]
    lines += [
        "",
        f"## Cells that currently survive same-station test (n_sites≥{args.min_pair_n}, p<0.05)",
        "",
    ]
    if survivors.empty:
        lines.append("- None at this threshold. Loosen buffer / increase labels, or treat as exploratory spatial pattern only.")
    else:
        for r in survivors.sort_values("mw_p").itertuples():
            direction = "faster decay than expected" if r.site_delta_med < 0 else "slower decay than expected"
            lines.append(
                f"- **#{int(r.rank)} {r.geo_tag}** ({r.lon:.2f}, {r.lat:.2f}): site-median Δβ={r.site_delta_med:+.4f} "
                f"({direction}); {100*r.site_delta_frac_neg:.0f}% of paired stations agree in sign; d={r.cohen_d:+.2f}."
            )
    lines += [
        "",
        "## Publication claim status",
        "- Spatial residual map + geologic tags: **done**.",
        "- Same-station survival: see above.",
        "- Still required before a discovery claim: independent Qc/tomography overlay on survivor cells, and one out-of-region replication.",
        "",
        "Figures: `same_station_validation.png`, `socal_structure_residuals.png`",
        "",
    ]
    (out / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False), flush=True)
    print(f"[done] {out / 'VALIDATION.md'}", flush=True)


if __name__ == "__main__":
    main()
