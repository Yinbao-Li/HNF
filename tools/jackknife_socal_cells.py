#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B2: source-bin / magnitude / year jackknife for SoCal discovery cells.

Extends Cascadia St Helens jackknife to Salton (#2) and ETR (#5).
"""

from __future__ import annotations

import argparse
import json
import re
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

GRID = 0.25
BUFFER_CELLS = 0.5
CELLS = {
    "salton": (-116.375, 33.375, "absorbing"),
    "etr": (-116.375, 33.875, "ringing"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="outputs/structure_residual_socal/traces_with_structure.csv")
    p.add_argument("--output-dir", default="outputs/structure_residual_socal")
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
        return float("nan")
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().to_numpy()
    ra = ranks[: len(a)].sum()
    na, nb = len(a), len(b)
    u = ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    sig = np.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (u - mu) / max(sig, 1e-9)
    return float(erfc(abs(z) / sqrt(2.0)))


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
        return {"n_sites": 0, "site_delta_med": float("nan"), "mw_p": float("nan"), "frac_sign": float("nan")}
    a_all = np.concatenate(paired_in)
    b_all = np.concatenate(paired_out)
    d = np.asarray(deltas, float)
    # sign: absorbing expects negative delta; ringing positive
    return {
        "n_sites": int(len(d)),
        "n_in": int(len(inside)),
        "n_in_paired": int(len(a_all)),
        "site_delta_med": float(np.nanmedian(d)),
        "frac_neg": float((d < 0).mean()),
        "frac_pos": float((d > 0).mean()),
        "mw_p": _mw_p(a_all, b_all),
    }


def jackknife_cell(df: pd.DataFrame, name: str, clon: float, clat: float, expect: str) -> dict:
    df = df.copy()
    df["year"] = df["trace_name"].map(_year)
    df["srcbin"] = (
        np.round(df["src_lon"] / 0.05) * 0.05
    ).round(5).astype(str) + "," + (np.round(df["src_lat"] / 0.05) * 0.05).round(5).astype(str)

    gx = np.floor(df["path_mid_lon"] / GRID) * GRID + 0.5 * GRID
    gy = np.floor(df["path_mid_lat"] / GRID) * GRID + 0.5 * GRID
    in_m = np.isclose(gx, clon) & np.isclose(gy, clat)
    vc = df.loc[in_m, "srcbin"].value_counts()
    years = sorted(y for y in df.loc[in_m, "year"].unique() if y > 0)

    tests = [{"test": "full", **same_station_delta(df, clon, clat), "dropped": ""}]

    for k in (1, 3, 5, 10):
        drop = set(vc.head(k).index.tolist())
        # Drop prolific in-cell sources only.
        mask_drop = in_m & df["srcbin"].isin(drop)
        sub = df.loc[~mask_drop].copy()
        tests.append({"test": f"drop_top{k}_srcbins", **same_station_delta(sub, clon, clat), "dropped": ",".join(sorted(drop)[:5])})

    for floor in (2.0, 2.5, 3.0):
        sub = df.loc[~(in_m & (df["mag"] < floor))].copy()
        tests.append({"test": f"mag_floor_{floor}", **same_station_delta(sub, clon, clat), "dropped": f"mag<{floor}"})

    for y in years:
        sub = df.loc[~(in_m & (df["year"] == y))].copy()
        tests.append({"test": f"drop_year_{y}", **same_station_delta(sub, clon, clat), "dropped": str(y)})

    # Cap traces per source bin inside cell.
    for cap in (5, 10, 20):
        keep_idx = []
        for _, g in df.loc[in_m].groupby("srcbin"):
            keep_idx.extend(g.head(cap).index.tolist())
        keep = set(keep_idx) | set(df.index[~in_m])
        sub = df.loc[sorted(keep)].copy()
        tests.append({"test": f"cap_srcbin_{cap}", **same_station_delta(sub, clon, clat), "dropped": f"cap={cap}"})

    # Stability: sign of site_delta_med matches expectation and p stays <0.05 when n_sites>=8.
    full = tests[0]
    sign_ok = []
    for t in tests:
        d = t.get("site_delta_med", float("nan"))
        if not np.isfinite(d) or t.get("n_sites", 0) < 8:
            continue
        ok = (d < 0) if expect == "absorbing" else (d > 0)
        sign_ok.append(bool(ok and t.get("mw_p", 1.0) < 0.05))
    frac_stable = float(np.mean(sign_ok)) if sign_ok else float("nan")
    verdict = "STABLE" if np.isfinite(frac_stable) and frac_stable >= 0.8 else "FRAGILE"
    return {
        "cell": name,
        "clon": clon,
        "clat": clat,
        "expect": expect,
        "n_srcbins_in_cell": int(vc.size),
        "n_years": int(len(years)),
        "full": full,
        "frac_tests_sign_and_p": frac_stable,
        "n_qualifying_tests": int(len(sign_ok)),
        "verdict": verdict,
        "tests": tests,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.traces)
    report = {"cells": {}, "protocol": "source-bin/mag/year jackknife; discovery threshold n_sites>=8, p<0.05"}
    for name, (clon, clat, expect) in CELLS.items():
        rec = jackknife_cell(df, name, clon, clat, expect)
        report["cells"][name] = rec
        print(
            f"[{name}] full Δ={rec['full']['site_delta_med']:+.4f} "
            f"n_sites={rec['full']['n_sites']} p={rec['full']['mw_p']:.3g} "
            f"stable={rec['frac_tests_sign_and_p']:.2f} → {rec['verdict']}",
            flush=True,
        )
    (out / "socal_cell_jackknife.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for name, rec in report["cells"].items():
        for t in rec["tests"]:
            rows.append({"cell": name, **{k: t.get(k) for k in ("test", "n_sites", "site_delta_med", "mw_p", "frac_neg", "frac_pos")}})
    pd.DataFrame(rows).to_csv(out / "socal_cell_jackknife.csv", index=False)
    print("Wrote", out / "socal_cell_jackknife.json")


if __name__ == "__main__":
    main()
