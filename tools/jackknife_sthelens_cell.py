#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source / year jackknife of the Cascadia St Helens–Cowlitz replica cell.

Checks whether same-station Δβ is driven by a few prolific St Helens sources
or a single year swarm.

Example
-------
  PYTHONPATH=. python tools/jackknife_sthelens_cell.py
"""

from __future__ import annotations

import argparse
import json
import re
from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_INK = "#1C1917"
C_MUTED = "#78716C"
BG = "#FFFEF9"
CELL = (-122.375, 46.375)
GRID = 0.25
BUFFER_CELLS = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="outputs/structure_residual_cascadia_volc/traces_with_structure.csv")
    p.add_argument("--output-dir", default="outputs/structure_residual_cascadia_volc")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _year(name: str) -> int:
    m = re.search(r"_(\d{4})\d{10}_", str(name))
    return int(m.group(1)) if m else -1


def _mw_p(a, b):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
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
    return float(z), float(erfc(abs(z) / sqrt(2.0)))


def same_station_delta(df: pd.DataFrame, clon: float, clat: float) -> dict:
    gx = np.floor(df["path_mid_lon"] / GRID) * GRID + 0.5 * GRID
    gy = np.floor(df["path_mid_lat"] / GRID) * GRID + 0.5 * GRID
    in_m = np.isclose(gx, clon) & np.isclose(gy, clat)
    buf = BUFFER_CELLS * GRID
    near = (np.abs(gx - clon) <= buf + 1e-9) & (np.abs(gy - clat) <= buf + 1e-9)
    inside = df.loc[in_m]
    ctrl = df.loc[~near]
    sites = sorted(set(inside["site"].astype(str)) & set(ctrl["site"].astype(str)))
    deltas, paired_in, paired_out = [], [], []
    for site in sites:
        a = inside.loc[inside["site"] == site, "beta_resid"].to_numpy(float)
        b = ctrl.loc[ctrl["site"] == site, "beta_resid"].to_numpy(float)
        if len(a) < 2 or len(b) < 2:
            continue
        deltas.append(float(np.nanmedian(a) - np.nanmedian(b)))
        paired_in.append(a)
        paired_out.append(b)
    if not deltas:
        return {"n_sites": 0, "n_in": 0, "site_delta_med": float("nan"), "frac_neg": float("nan"), "mw_p": float("nan")}
    a_all = np.concatenate(paired_in)
    b_all = np.concatenate(paired_out)
    _, p = _mw_p(a_all, b_all)
    d = np.asarray(deltas, float)
    return {
        "n_sites": int(len(d)),
        "n_in": int(len(inside)),
        "n_in_paired": int(len(a_all)),
        "site_delta_med": float(np.nanmedian(d)),
        "frac_neg": float((d < 0).mean()),
        "mw_p": p,
        "pooled_delta": float(np.nanmedian(a_all) - np.nanmedian(b_all)),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.traces)
    df["year"] = df["trace_name"].map(_year)
    df["srcbin"] = (
        np.round(df["src_lon"] / 0.05) * 0.05
    ).round(5).astype(str) + "," + (np.round(df["src_lat"] / 0.05) * 0.05).round(5).astype(str)

    gx = np.floor(df["path_mid_lon"] / GRID) * GRID + 0.5 * GRID
    gy = np.floor(df["path_mid_lat"] / GRID) * GRID + 0.5 * GRID
    in_m = np.isclose(gx, CELL[0]) & np.isclose(gy, CELL[1])
    vc = df.loc[in_m, "srcbin"].value_counts()
    years = sorted(y for y in df.loc[in_m, "year"].unique() if y > 0)

    tests = []
    tests.append({"test": "full", **same_station_delta(df, *CELL), "dropped": ""})

    for k in (1, 3, 5, 10):
        drop = set(vc.head(k).index)
        sub = df[~((in_m) & (df["srcbin"].isin(drop)))].copy()
        # keep controls intact; only thin the in-cell traces from prolific sources
        rec = same_station_delta(sub, *CELL)
        tests.append({"test": f"drop_top{k}_srcbins", **rec, "dropped": ",".join(drop)})

    rng = np.random.default_rng(0)
    inside_idx = df.index[in_m]
    for cap in (40, 20, 10):
        chosen = []
        for _, grp in df.loc[inside_idx].groupby("srcbin"):
            take = min(len(grp), cap)
            chosen.extend(rng.choice(grp.index.to_numpy(), size=take, replace=False).tolist())
        sub = pd.concat([df.loc[~in_m], df.loc[chosen]], axis=0)
        rec = same_station_delta(sub, *CELL)
        tests.append({"test": f"cap{cap}_per_srcbin", **rec, "dropped": f"cap={cap}"})

    for y in years:
        sub = df[df["year"] != y]
        rec = same_station_delta(sub, *CELL)
        tests.append({"test": f"drop_year_{y}", **rec, "dropped": str(y)})

    # magnitude cuts
    for mag0 in (0.5, 1.0, 1.5):
        sub = df[~(in_m & (df["mag"] < mag0))]
        rec = same_station_delta(sub, *CELL)
        tests.append({"test": f"in_cell_mag>={mag0}", **rec, "dropped": f"M<{mag0}"})

    out = pd.DataFrame(tests)
    out_dir = Path(args.output_dir)
    out.to_csv(out_dir / "sthelens_source_jackknife.csv", index=False)

    # figure: main stress tests
    focus = out[out["test"].isin(
        ["full", "drop_top1_srcbins", "drop_top5_srcbins", "drop_top10_srcbins", "cap20_per_srcbin", "cap10_per_srcbin", "in_cell_mag>=1.0", "in_cell_mag>=1.5"]
    ) | out["test"].str.startswith("drop_year_")].copy()
    fig, ax = plt.subplots(figsize=(10.8, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axhline(0, color=C_MUTED, lw=0.8, ls="--")
    xs = np.arange(len(focus))
    cols = ["#B91C1C" if v < 0 else "#1D4ED8" for v in focus["site_delta_med"]]
    ax.bar(xs, focus["site_delta_med"], color=cols, width=0.72, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(focus["test"], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(r"same-station median $\Delta\beta$ (St Helens cell − control)", fontsize=8)
    ax.set_title("St Helens–Cowlitz replica: source / year jackknife", color=C_INK, fontsize=10)
    for i, r in enumerate(focus.itertuples()):
        ax.text(i, r.site_delta_med - 0.001, f"st={int(r.n_sites)}\np≈{r.mw_p:.1g}", ha="center", va="top", fontsize=6, color=C_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "sthelens_source_jackknife.png", dpi=args.dpi, facecolor=BG)
    fig.savefig(out_dir / "sthelens_source_jackknife.pdf", dpi=args.dpi, facecolor=BG)
    plt.close(fig)

    full = out.loc[out["test"] == "full"].iloc[0]
    survivors = out[
        (out["test"] != "full")
        & (out["n_sites"] >= 6)
        & (out["site_delta_med"] < 0)
        & (out["frac_neg"] >= 0.6)
    ]
    report = {
        "cell": {"lon": CELL[0], "lat": CELL[1], "name": "SW Washington / Mt St Helens–Cowlitz"},
        "full": full.drop(labels=["dropped"]).to_dict(),
        "n_tests": int(len(out)),
        "n_sign_stable": int(len(survivors)),
        "worst_site_delta": float(out["site_delta_med"].max()),
        "best_site_delta": float(out["site_delta_med"].min()),
        "year_loo_all_negative": bool(
            (out.loc[out["test"].str.startswith("drop_year_"), "site_delta_med"] < 0).all()
        ),
        "top_srcbin_frac": {
            "top1": float(vc.iloc[0] / in_m.sum()),
            "top5": float(vc.head(5).sum() / in_m.sum()),
            "top10": float(vc.head(10).sum() / in_m.sum()),
        },
        "verdict": (
            "STABLE"
            if bool((out.loc[out["test"].str.startswith("drop_year_"), "site_delta_med"] < 0).all())
            and float(out.loc[out["test"] == "drop_top5_srcbins", "site_delta_med"].iloc[0]) < 0
            and float(out.loc[out["test"] == "cap20_per_srcbin", "site_delta_med"].iloc[0]) < 0
            else "FRAGILE"
        ),
    }
    (out_dir / "sthelens_source_jackknife.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out[["test", "n_sites", "n_in", "site_delta_med", "frac_neg", "mw_p"]].to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] {out_dir / 'sthelens_source_jackknife.png'}", flush=True)


if __name__ == "__main__":
    main()
