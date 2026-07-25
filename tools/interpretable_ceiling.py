#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretable ceiling: site terms + mag-type models + shape×strength taxonomy.

Takes the feature table from ``reclassify_causal_physics.py`` (no GPU needed) and:

1. Fits a Richter base ``M ~ logA + logD`` then additive **station / network**
   residual corrections (classic site terms — still a lookup table).
2. Trains **separate** models for ``ml`` and ``md`` (mixed scales were a
   major residual source).
3. Relabels every trace with a 2-D interpretable tag
   ``{shape}×{strength}`` instead of a flat k-means id.
4. Reports magnitude R²/MAE and clustering discrimination via region Cramér's V
   and coda path-residual effect sizes (not silhouette).

Example:
  PYTHONPATH=. python tools/interpretable_ceiling.py \\
    --traces outputs/causal_reclass_run28/traces.csv \\
    --output-dir outputs/interpretable_ceiling_run28
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpretable mag ceiling + 2D taxonomy")
    p.add_argument("--traces", default="outputs/causal_reclass_run28/traces.csv")
    p.add_argument("--output-dir", default="outputs/interpretable_ceiling_run28")
    p.add_argument("--min-station-events", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def richter_coef(log_a: np.ndarray, log_d: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.column_stack([log_a, log_d, np.ones(len(y))])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def apply_richter(coef: np.ndarray, log_a: np.ndarray, log_d: np.ndarray) -> np.ndarray:
    return coef[0] * log_a + coef[1] * log_d + coef[2]


def cv_site_corrected(
    df: pd.DataFrame,
    *,
    min_station: int,
    seed: int,
    n_folds: int = 5,
) -> dict:
    """Richter + additive station (fallback network) residual correction, 5-fold CV."""
    y = df["mag"].to_numpy(dtype=float)
    log_a = df["log_peak_amp"].to_numpy(dtype=float)
    log_d = np.log10(df["dist_km"].to_numpy(dtype=float) + 1.0)
    stations = df["station"].astype(str).to_numpy()
    networks = df["network"].astype(str).to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    preds = np.full(n, np.nan)
    site_tables = []

    for fi in range(n_folds):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        coef = richter_coef(log_a[tr], log_d[tr], y[tr])
        base_tr = apply_richter(coef, log_a[tr], log_d[tr])
        resid_tr = y[tr] - base_tr

        st_sums: dict[str, list[float]] = {}
        net_sums: dict[str, list[float]] = {}
        for s, net, r in zip(stations[tr], networks[tr], resid_tr):
            st_sums.setdefault(s, []).append(float(r))
            net_sums.setdefault(net, []).append(float(r))
        st_mean = {s: float(np.mean(v)) for s, v in st_sums.items() if len(v) >= min_station}
        net_mean = {s: float(np.mean(v)) for s, v in net_sums.items() if len(v) >= min_station}
        site_tables.append({"fold": fi, "n_stations": len(st_mean), "n_networks": len(net_mean)})

        base_te = apply_richter(coef, log_a[te], log_d[te])
        corr = np.array(
            [st_mean.get(stations[i], net_mean.get(networks[i], 0.0)) for i in te],
            dtype=float,
        )
        preds[te] = base_te + corr

    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return {
        "r2": round(1.0 - ss_res / ss_tot, 3),
        "mae": round(float(np.abs(y - preds).mean()), 3),
        "rmse": round(float(np.sqrt(((y - preds) ** 2).mean())), 3),
        "n": int(n),
        "site_tables": site_tables,
        "predictions": preds,
        "residuals": y - preds,
    }


def richter_only_cv(df: pd.DataFrame, seed: int = 0) -> dict:
    y = df["mag"].to_numpy(dtype=float)
    log_a = df["log_peak_amp"].to_numpy(dtype=float)
    log_d = np.log10(df["dist_km"].to_numpy(dtype=float) + 1.0)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)
    preds = np.full(n, np.nan)
    for fi in range(5):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(5) if j != fi])
        coef = richter_coef(log_a[tr], log_d[tr], y[tr])
        preds[te] = apply_richter(coef, log_a[te], log_d[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return {
        "r2": round(1.0 - ss_res / ss_tot, 3),
        "mae": round(float(np.abs(y - preds).mean()), 3),
        "rmse": round(float(np.sqrt(((y - preds) ** 2).mean())), 3),
        "n": int(n),
    }


def assign_shape(row: pd.Series) -> str:
    peaks = float(row["n_rho_peaks"])
    coda = float(row["coda_slope"])
    onset = float(row["onset_sharp"])
    if peaks >= 0.7:
        return "multipath"
    if onset >= 0.8 and coda <= -0.20:
        return "impulsive_fastQ"
    if onset < 0.65:
        return "emergent"
    if coda > -0.12:
        return "slow_coda"
    return "standard"


def assign_strength(reduced: float, q33: float, q66: float) -> str:
    if reduced >= q66:
        return "strong"
    if reduced >= q33:
        return "mid"
    return "weak"


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    table = table.astype(np.float64)
    if table.sum() == 0:
        return 0.0, 0.0
    row = table.sum(1, keepdims=True)
    col = table.sum(0, keepdims=True)
    exp = row @ col / table.sum()
    mask = exp > 0
    chi2 = float(((table[mask] - exp[mask]) ** 2 / exp[mask]).sum())
    n = float(table.sum())
    r, c = table.shape
    v = float(np.sqrt(chi2 / (n * max(min(r - 1, c - 1), 1))))
    return chi2, v


def region_association(df: pd.DataFrame, label_col: str, region_col: str = "path_region") -> dict:
    vc = df[region_col].value_counts()
    keep = list(vc.head(12).index)
    labels = sorted(df[label_col].dropna().unique())
    table = np.zeros((len(keep), len(labels)), dtype=np.int64)
    for i, r in enumerate(keep):
        for j, lab in enumerate(labels):
            table[i, j] = int(((df[region_col] == r) & (df[label_col] == lab)).sum())
    chi2, v = cramers_v(table)
    fracs = {}
    for i, r in enumerate(keep):
        tot = max(int(table[i].sum()), 1)
        fracs[r] = {str(labels[j]): round(float(table[i, j]) / tot, 3) for j in range(len(labels))}
        fracs[r]["n"] = int(table[i].sum())
    controlled = {}
    bins = pd.cut(df["dist_km"], bins=[0, 50, 150, 400, 5000], labels=["0-50", "50-150", "150-400", "400+"])
    for b in ["0-50", "50-150"]:
        sub = df[bins == b]
        if len(sub) < 40:
            continue
        vc2 = sub[region_col].value_counts()
        keep2 = list(vc2.head(8).index)
        labs2 = sorted(sub[label_col].dropna().unique())
        t2 = np.zeros((len(keep2), len(labs2)), dtype=np.int64)
        for i, r in enumerate(keep2):
            for j, lab in enumerate(labs2):
                t2[i, j] = int(((sub[region_col] == r) & (sub[label_col] == lab)).sum())
        c2, v2 = cramers_v(t2)
        controlled[b] = {"n": int(len(sub)), "chi2": round(c2, 1), "cramers_v": round(v2, 3)}
    return {"chi2": round(chi2, 1), "cramers_v": round(v, 3), "fractions": fracs, "distance_controlled": controlled}


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two residual groups."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    pooled = np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.traces)
    need = ["mag", "dist_km", "log_peak_amp", "station", "network", "mag_type",
            "coda_slope", "onset_sharp", "n_rho_peaks", "reduced_amp", "ps_gap"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"missing column {c} in {args.traces}")

    use = (
        df["mag"].notna()
        & df["dist_km"].notna()
        & (df["dist_km"] > 0)
        & df["log_peak_amp"].notna()
        & df["station"].notna()
    )
    d = df.loc[use].copy().reset_index(drop=True)

    # path region if absent
    if "path_region" not in d.columns:
        if {"src_lat", "src_lon", "rcv_lat", "rcv_lon"}.issubset(d.columns):
            mid_lat = 0.5 * (d["src_lat"] + d["rcv_lat"])
            mid_lon = 0.5 * (d["src_lon"] + d["rcv_lon"])
            d["path_region"] = [
                f"{int(np.floor(a / 5) * 5):+d}/{int(np.floor(b / 5) * 5):+d}"
                if np.isfinite(a) and np.isfinite(b)
                else "unknown"
                for a, b in zip(mid_lat, mid_lon)
            ]
        else:
            d["path_region"] = "unknown"

    # ---- magnitude: baselines + ceiling ----
    mag_report: dict = {
        "richter_all": richter_only_cv(d, seed=args.seed),
        "richter_site_all": {k: v for k, v in cv_site_corrected(
            d, min_station=args.min_station_events, seed=args.seed
        ).items() if k not in {"predictions", "residuals", "site_tables"}},
    }
    full_site = cv_site_corrected(d, min_station=args.min_station_events, seed=args.seed)
    d["mag_pred_site"] = full_site["predictions"]
    d["mag_resid_site"] = full_site["residuals"]

    # per mag-type models, then pool predictions for overall R²
    pooled_pred = np.full(len(d), np.nan)
    covered = np.zeros(len(d), dtype=bool)
    per_type = {}
    for mt in ["ml", "md", "mb"]:
        sub_idx = np.where(d["mag_type"].to_numpy() == mt)[0]
        if len(sub_idx) < 40:
            continue
        sub = d.iloc[sub_idx].reset_index(drop=True)
        base = richter_only_cv(sub, seed=args.seed)
        site = cv_site_corrected(sub, min_station=max(2, args.min_station_events - 1), seed=args.seed)
        per_type[mt] = {
            "richter": base,
            "richter_site": {k: site[k] for k in ("r2", "mae", "rmse", "n")},
        }
        pooled_pred[sub_idx] = site["predictions"]
        covered[sub_idx] = True

    # rare / skipped types: fall back to global site model
    if (~covered).any():
        pooled_pred[~covered] = full_site["predictions"][~covered]
    y = d["mag"].to_numpy(float)
    ss_res = float(((y - pooled_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    mag_report["per_type"] = per_type
    mag_report["stratified_site_pooled"] = {
        "r2": round(1.0 - ss_res / ss_tot, 3),
        "mae": round(float(np.abs(y - pooled_pred).mean()), 3),
        "rmse": round(float(np.sqrt(((y - pooled_pred) ** 2).mean())), 3),
        "n": int(len(y)),
        "note": "ml/md/mb each get their own Richter+site model; rare types use global site model",
    }
    d["mag_pred_stratified"] = pooled_pred
    d["mag_resid_stratified"] = y - pooled_pred

    # final global site table (fit on all data) for export — interpretable artifact
    coef = richter_coef(
        d["log_peak_amp"].to_numpy(float),
        np.log10(d["dist_km"].to_numpy(float) + 1.0),
        y,
    )
    base_all = apply_richter(
        coef,
        d["log_peak_amp"].to_numpy(float),
        np.log10(d["dist_km"].to_numpy(float) + 1.0),
    )
    resid_all = y - base_all
    st_table = (
        pd.DataFrame({"station": d["station"], "network": d["network"], "resid": resid_all})
        .groupby(["network", "station"], as_index=False)["resid"]
        .agg(site_term="mean", n_events="count")
    )
    st_table = st_table[st_table["n_events"] >= args.min_station_events].sort_values("site_term")
    st_table.to_csv(out / "site_terms.csv", index=False)
    net_table = (
        pd.DataFrame({"network": d["network"], "resid": resid_all})
        .groupby("network", as_index=False)["resid"]
        .agg(network_term="mean", n_events="count")
        .sort_values("network_term")
    )
    net_table.to_csv(out / "network_terms.csv", index=False)

    mag_report["richter_equation"] = {
        "form": "M ≈ a*log10(A) + b*log10(D+1) + c + site_term(station)",
        "a": round(float(coef[0]), 4),
        "b": round(float(coef[1]), 4),
        "c": round(float(coef[2]), 4),
        "n_site_terms": int(len(st_table)),
        "n_network_terms": int(len(net_table)),
    }

    # ---- 2-D taxonomy: shape × strength ----
    q33, q66 = d["reduced_amp"].quantile(0.33), d["reduced_amp"].quantile(0.66)
    d["shape"] = d.apply(assign_shape, axis=1)
    d["strength"] = [assign_strength(v, q33, q66) for v in d["reduced_amp"]]
    d["tax2d"] = d["shape"] + "×" + d["strength"]

    # coda path residual (distance-detrended) for structure
    ok = d["coda_slope"].notna() & d["dist_km"].notna()
    ld = np.log10(d.loc[ok, "dist_km"].to_numpy(float) + 1.0)
    cs = d.loc[ok, "coda_slope"].to_numpy(float)
    A = np.column_stack([ld, np.ones(len(ld))])
    ccoef, _, _, _ = np.linalg.lstsq(A, cs, rcond=None)
    d.loc[ok, "coda_path_residual"] = cs - A @ ccoef

    # discrimination report
    disc = {
        "shape_vs_region": region_association(d, "shape"),
        "strength_vs_region": region_association(d, "strength"),
        "tax2d_vs_region": region_association(d, "tax2d"),
        "shape_counts": d["shape"].value_counts().to_dict(),
        "strength_counts": d["strength"].value_counts().to_dict(),
        "tax2d_counts": d["tax2d"].value_counts().to_dict(),
        "reduced_amp_terciles": {"q33": round(float(q33), 3), "q66": round(float(q66), 3)},
    }

    # path residual effect sizes by shape (vs rest)
    shape_effects = []
    for sh in sorted(d["shape"].unique()):
        a = d.loc[d["shape"] == sh, "coda_path_residual"].to_numpy(float)
        b = d.loc[d["shape"] != sh, "coda_path_residual"].to_numpy(float)
        shape_effects.append({
            "shape": sh,
            "n": int((d["shape"] == sh).sum()),
            "coda_residual_mean": round(float(np.nanmean(a)), 4),
            "cohens_d_vs_rest": round(effect_size(a, b), 3),
            "mag_mean": round(float(d.loc[d["shape"] == sh, "mag"].mean()), 2),
            "top_regions": d.loc[d["shape"] == sh, "path_region"].value_counts().head(3).to_dict(),
        })
    disc["shape_path_effects"] = shape_effects

    # mag separation by strength / tax2d
    mag_sep = {}
    for col in ["strength", "shape", "tax2d"]:
        rows = []
        for lab, g in d.groupby(col):
            rows.append({
                "label": str(lab),
                "n": int(len(g)),
                "mag_mean": round(float(g["mag"].mean()), 2),
                "mag_std": round(float(g["mag"].std()), 2),
            })
        mag_sep[col] = sorted(rows, key=lambda r: -r["mag_mean"])
    disc["magnitude_by_label"] = mag_sep

    report = {
        "n_traces": int(len(d)),
        "magnitude": mag_report,
        "discrimination": disc,
        "headline": {
            "richter_r2": mag_report["richter_all"]["r2"],
            "site_r2": mag_report["richter_site_all"]["r2"],
            "stratified_site_r2": mag_report["stratified_site_pooled"]["r2"],
            "stratified_site_mae": mag_report["stratified_site_pooled"]["mae"],
            "shape_region_V": disc["shape_vs_region"]["cramers_v"],
            "shape_region_V_0_50km": disc["shape_vs_region"]["distance_controlled"].get("0-50", {}).get("cramers_v"),
            "tax2d_region_V": disc["tax2d_vs_region"]["cramers_v"],
            "ceiling_note": (
                "Interpretable single-station ceiling on this STEAD slice is ~0.83–0.85 R². "
                "0.95 is not a realistic KPI without multi-station / unified magnitude scale."
            ),
        },
    }
    (out / "ceiling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    d.to_csv(out / "traces_labeled.csv", index=False)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        # 1. predicted vs true (stratified site)
        ax = axes[0, 0]
        ax.scatter(d["mag"], d["mag_pred_stratified"], s=12, alpha=0.45, c="steelblue")
        lims = [0, max(6, d["mag"].max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("catalog mag")
        ax.set_ylabel("predicted mag")
        ax.set_title(
            f"Stratified Richter+site  R²={mag_report['stratified_site_pooled']['r2']}  "
            f"MAE={mag_report['stratified_site_pooled']['mae']}"
        )

        # 2. mag by strength
        ax = axes[0, 1]
        order = ["weak", "mid", "strong"]
        data = [d.loc[d["strength"] == s, "mag"].to_numpy() for s in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_ylabel("magnitude")
        ax.set_title("Strength axis (reduced_amp terciles)")

        # 3. mag by shape
        ax = axes[1, 0]
        shapes = sorted(d["shape"].unique())
        data = [d.loc[d["shape"] == s, "mag"].to_numpy() for s in shapes]
        ax.boxplot(data, labels=shapes, showfliers=False)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_ylabel("magnitude")
        ax.set_title(f"Shape axis (region V={disc['shape_vs_region']['cramers_v']})")

        # 4. coda path residual by shape
        ax = axes[1, 1]
        data = [d.loc[d["shape"] == s, "coda_path_residual"].dropna().to_numpy() for s in shapes]
        ax.boxplot(data, labels=shapes, showfliers=False)
        ax.axhline(0, color="k", lw=0.6)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_ylabel("coda path residual")
        ax.set_title("Structure proxy by shape (distance-detrended)")

        fig.tight_layout()
        fig.savefig(out / "ceiling_overview.png", dpi=130)
        print(f"[ceiling] wrote {out / 'ceiling_overview.png'}", flush=True)
    except Exception as exc:
        print(f"[ceiling] plot skipped: {exc}", flush=True)

    print(json.dumps(report["headline"], indent=2), flush=True)
    print(json.dumps({"magnitude": mag_report, "shape_effects": shape_effects}, indent=2), flush=True)


if __name__ == "__main__":
    main()
