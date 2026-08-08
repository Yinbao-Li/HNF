#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical coda baseline vs PNF-timed discovery (SoCal incremental value).

Pipelines compared
------------------
1. PNF-timed: existing coda_path_residual / beta_resid (PNF S picks + raw envelope).
2. Catalog-timed classical: STEAD catalog S sample → raw-envelope coda slope →
   same dist-only then structure residualization.
3. Incremental ΔR²: classical features ± PNF latents (n_rho_peaks, facies one-hots)
   predicting beta_resid (and catalog beta_resid_classical).

Writes outputs/structure_residual_socal/pnf_vs_classical_coda/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEAD_DIR
from tools.analyze_structure_residual_anomalies import _design_matrix, ridge_fit

GRID = 0.25
CELLS = {
    2: (-116.375, 33.375, "Salton / S.SAF–Imperial"),
    5: (-116.375, 33.875, "E. Transverse Ranges"),
}
SHAPE_ORDER = ["multipath", "impulsive_fastQ", "emergent", "slow_coda", "standard"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traces-structure", default="outputs/structure_residual_socal/traces_with_structure.csv")
    p.add_argument("--traces-labeled", default="outputs/shape_labels_expanded/socal/traces_labeled.csv")
    p.add_argument("--panel", default="outputs/shape_labels_expanded/socal/panel_selection.csv")
    p.add_argument("--stead-dir", default=str(STEAD_DIR))
    p.add_argument("--max-traces", type=int, default=0, help="0 = all overlapping traces")
    p.add_argument("--ridge", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output-dir", default="outputs/structure_residual_socal/pnf_vs_classical_coda")
    return p.parse_args()


def _env(wave: np.ndarray) -> np.ndarray:
    e = np.mean(np.square(wave.astype(np.float64)), axis=1)
    return np.convolve(e, np.ones(21) / 21.0, mode="same")


def coda_slope_at_s(wave: np.ndarray, s_sec: float, coda_sec: float = 8.0, window_sec: float = 60.0) -> float:
    e = _env(wave)
    n = e.size
    t = np.linspace(0.0, window_sec, n)
    lo = float(s_sec)
    hi = min(window_sec, lo + coda_sec)
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


def dist_detrend(slope: np.ndarray, dist_km: np.ndarray) -> np.ndarray:
    x = np.log10(np.asarray(dist_km, float) + 1.0)
    y = np.asarray(slope, float)
    m = np.isfinite(x) & np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=float)
    if m.sum() < 10:
        return out
    A = np.column_stack([x[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    out[m] = y[m] - A @ coef
    return out


def oos_r2(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int, ridge: float = 1.0) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    pred = np.full(n, np.nan)
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        if len(tr) < 20 or len(te) < 5:
            continue
        Xt, yt = X[tr], y[tr]
        w = ridge_fit(Xt, yt, ridge)
        pred[te] = X[te] @ w
    m = np.isfinite(pred)
    if m.sum() < 20:
        return float("nan")
    ss_res = float(np.sum((y[m] - pred[m]) ** 2))
    ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def one_hot_shape(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    mats = []
    names = []
    for sh in SHAPE_ORDER[:-1]:  # drop last for full rank
        mats.append((series.astype(str) == sh).astype(float).to_numpy())
        names.append(f"shape_{sh}")
    return np.column_stack(mats) if mats else np.zeros((len(series), 0)), names


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    struct = pd.read_csv(args.traces_structure)
    labeled = pd.read_csv(args.traces_labeled)
    panel = pd.read_csv(
        args.panel,
        usecols=["trace_name", "chunk", "p_arrival_sample", "s_arrival_sample"],
    )
    feat_cols = ["trace_name", "coda_slope", "onset_sharp", "n_rho_peaks", "ps_gap"]
    feat_cols = [c for c in feat_cols if c in labeled.columns]
    df = struct.merge(labeled[feat_cols], on="trace_name", how="inner", suffixes=("", "_lab"))
    df = df.merge(panel, on="trace_name", how="inner")
    df = df.dropna(subset=["beta_resid", "s_arrival_sample", "chunk", "dist_km"])
    if args.max_traces and len(df) > args.max_traces:
        rng = np.random.default_rng(args.seed)
        # keep all cell traces, sample the rest
        df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])
        must = pd.Series(False, index=df.index)
        for lon, lat, _ in CELLS.values():
            must |= np.isclose(df["gx"], lon) & np.isclose(df["gy"], lat)
        idx_must = df.index[must].to_numpy()
        idx_rest = df.index[~must].to_numpy()
        need = max(args.max_traces - len(idx_must), 0)
        take = rng.choice(idx_rest, size=min(need, len(idx_rest)), replace=False) if need else np.array([], dtype=int)
        df = df.loc[np.concatenate([idx_must, take])].copy()
    else:
        df["gx"], df["gy"] = zip(*[_cell_key(a, b) for a, b in zip(df["path_mid_lon"], df["path_mid_lat"])])

    print(f"[b1] computing catalog-timed classical coda on {len(df)} traces", flush=True)
    stead = Path(args.stead_dir)
    handles: dict[int, h5py.File] = {}

    def h5(c: int):
        if c not in handles:
            handles[c] = h5py.File(stead / f"chunk{c}_eofextract" / f"chunk{c}.hdf5", "r")
        return handles[c]

    catalog_slopes = []
    for i, row in enumerate(df.itertuples()):
        try:
            w = h5(int(row.chunk))["data"][str(row.trace_name)][()]
            # STEAD native 100 Hz sample index → seconds
            s_sec = float(row.s_arrival_sample) / 100.0
            catalog_slopes.append(coda_slope_at_s(w, s_sec))
        except Exception:
            catalog_slopes.append(np.nan)
        if (i + 1) % 1000 == 0:
            print(f"[b1] {i+1}/{len(df)}", flush=True)
    for h in handles.values():
        h.close()

    df["classical_coda_slope"] = catalog_slopes
    df = df.dropna(subset=["classical_coda_slope"]).copy()
    print(f"[b1] kept {len(df)} with finite catalog-timed slope", flush=True)

    df["classical_path_residual"] = dist_detrend(df["classical_coda_slope"].to_numpy(), df["dist_km"].to_numpy())

    # Structure residualization on classical path residual (same X as PNF beta_resid pipeline)
    X, _, keep = _design_matrix(df)
    y_c = df["classical_path_residual"].to_numpy(float)
    # _design_matrix may drop rows via keep mask
    if keep is not None and len(keep) == len(df):
        # some versions return boolean mask; others return all-True via index
        pass
    # Use full design as in analyze_structure_residual_anomalies
    X, colnames, site_codes = _design_matrix(df)
    w = ridge_fit(X, y_c, args.ridge)
    df["beta_resid_classical"] = y_c - X @ w

    # Correlations
    r_pnf_vs_cat = float(np.corrcoef(df["coda_slope"], df["classical_coda_slope"])[0, 1]) if "coda_slope" in df.columns else float("nan")
    r_beta = float(np.corrcoef(df["beta_resid"], df["beta_resid_classical"])[0, 1])
    r_path = float(np.corrcoef(df["coda_path_residual"], df["classical_path_residual"])[0, 1])

    cell_rows = []
    for rank, (lon, lat, name) in CELLS.items():
        m = np.isclose(df["gx"], lon) & np.isclose(df["gy"], lat)
        # same-station: stations that appear both in cell and outside
        sites_cell = set(df.loc[m, "site"].astype(str))
        for site in sorted(sites_cell):
            in_c = m & (df["site"].astype(str) == site)
            out_c = (~m) & (df["site"].astype(str) == site)
            if in_c.sum() < 2 or out_c.sum() < 2:
                continue
            for col, tag in (("beta_resid", "pnf"), ("beta_resid_classical", "classical")):
                dlt = float(df.loc[in_c, col].median() - df.loc[out_c, col].median())
                cell_rows.append({"rank": rank, "cell": name, "site": site, "pipeline": tag, "delta_beta": dlt, "n_in": int(in_c.sum()), "n_out": int(out_c.sum())})
    cell_df = pd.DataFrame(cell_rows)
    cell_summary = []
    if len(cell_df):
        for (rank, cell, pipe), g in cell_df.groupby(["rank", "cell", "pipeline"]):
            cell_summary.append(
                {
                    "rank": int(rank),
                    "cell": cell,
                    "pipeline": pipe,
                    "n_sites": int(g["site"].nunique()),
                    "median_delta_beta": float(g["delta_beta"].median()),
                    "mean_delta_beta": float(g["delta_beta"].mean()),
                    "frac_neg": float((g["delta_beta"] < 0).mean()),
                    "frac_pos": float((g["delta_beta"] > 0).mean()),
                }
            )

    # ΔR²: predict PNF beta_resid from classical vs classical+PNF latents
    y = df["beta_resid"].to_numpy(float)
    classical = np.column_stack(
        [
            df["classical_coda_slope"].to_numpy(float),
            df["classical_path_residual"].to_numpy(float),
            np.log10(df["dist_km"].to_numpy(float) + 1.0),
        ]
    )
    oh, oh_names = one_hot_shape(df["shape"])
    pnf_lat = [df["n_rho_peaks"].to_numpy(float)]
    pnf_names = ["n_rho_peaks"]
    if "onset_sharp" in df.columns:
        pnf_lat.append(df["onset_sharp"].to_numpy(float))
        pnf_names.append("onset_sharp")
    pnf = np.column_stack(pnf_lat + ([oh] if oh.size else []))
    pnf_names = pnf_names + oh_names

    # standardize columns for ridge stability
    def _z(A):
        mu = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return (A - mu) / sd

    X_c = _z(classical)
    X_cp = _z(np.column_stack([classical, pnf]))
    r2_c = oos_r2(X_c, y, args.n_folds, args.seed, ridge=1.0)
    r2_cp = oos_r2(X_cp, y, args.n_folds, args.seed + 1, ridge=1.0)
    # also predict classical beta from PNF-only latents (facies+peaks)
    X_p = _z(pnf)
    y_c = df["beta_resid_classical"].to_numpy(float)
    r2_pnf_on_classical_beta = oos_r2(X_p, y_c, args.n_folds, args.seed + 2, ridge=1.0)
    r2_classical_on_classical_beta = oos_r2(X_c, y_c, args.n_folds, args.seed + 3, ridge=1.0)

    summary = {
        "n_traces": int(len(df)),
        "corr_coda_slope_pnf_vs_catalog": r_pnf_vs_cat,
        "corr_path_residual_pnf_vs_classical": r_path,
        "corr_beta_resid_pnf_vs_classical": r_beta,
        "oos_r2_predict_pnf_beta": {
            "classical_only": r2_c,
            "classical_plus_pnf_latents": r2_cp,
            "delta_r2": float(r2_cp - r2_c) if np.isfinite(r2_c) and np.isfinite(r2_cp) else float("nan"),
            "classical_features": ["classical_coda_slope", "classical_path_residual", "log10_dist"],
            "pnf_latent_features": pnf_names,
        },
        "oos_r2_predict_classical_beta": {
            "classical_only": r2_classical_on_classical_beta,
            "pnf_latents_only": r2_pnf_on_classical_beta,
        },
        "cell_same_station": cell_summary,
        "note": (
            "Catalog-timed classical coda uses STEAD s_arrival_sample/100 and raw envelope; "
            "structure residualization mirrors analyze_structure_residual_anomalies ridge."
        ),
    }

    df[
        [
            "trace_name",
            "site",
            "shape",
            "gx",
            "gy",
            "beta_resid",
            "beta_resid_classical",
            "coda_path_residual",
            "classical_path_residual",
            "classical_coda_slope",
            "n_rho_peaks",
        ]
        + (["coda_slope"] if "coda_slope" in df.columns else [])
    ].to_csv(out / "traces_pnf_vs_classical.csv", index=False)
    cell_df.to_csv(out / "same_station_deltas.csv", index=False)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# PNF vs classical catalog-timed coda",
        "",
        f"- n = **{summary['n_traces']}**",
        f"- corr(coda_slope PNF-timed, catalog-timed) = **{r_pnf_vs_cat:.3f}**",
        f"- corr(path residual) = **{r_path:.3f}**",
        f"- corr(β_res) = **{r_beta:.3f}**",
        f"- OOS R²(β_res ← classical) = **{r2_c:.4f}**",
        f"- OOS R²(β_res ← classical+PNF latents) = **{r2_cp:.4f}**",
        f"- ΔR² = **{summary['oos_r2_predict_pnf_beta']['delta_r2']:.4f}**",
        "",
        "## Same-station cell medians",
    ]
    for row in cell_summary:
        md.append(
            f"- cell {row['rank']} {row['cell']} [{row['pipeline']}]: "
            f"median Δβ={row['median_delta_beta']:+.4f} (n_sites={row['n_sites']})"
        )
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[b1] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
