#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretable ceiling: site terms + mag-type models + shape×strength taxonomy.

Takes the feature table from ``reclassify_causal_physics.py`` (no GPU needed) and:

1. Fits Richter ``M ~ logA + logD`` then additive **station / network** site terms.
2. Extends with interpretable covariates (depth, SNR, coda path residual) under CV.
3. Trains **separate** models for ``ml`` / ``md`` and pools predictions.
4. Relabels every trace with ``{shape}×{strength}`` (data-driven shape thresholds).
5. Reports magnitude R²/MAE and geography Cramér's V + path-residual effect sizes.

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


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def richter_coef(log_a: np.ndarray, log_d: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.column_stack([log_a, log_d, np.ones(len(y))])
    return _lstsq(x, y)


def apply_richter(coef: np.ndarray, log_a: np.ndarray, log_d: np.ndarray) -> np.ndarray:
    return coef[0] * log_a + coef[1] * log_d + coef[2]


def design_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    mats = []
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=float(np.nanmedian(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.0)
        mats.append(v)
    mats.append(np.ones(len(df)))
    return np.column_stack(mats)


def cv_linear(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    n_folds: int = 5,
    site: bool = False,
    min_station: int = 3,
) -> dict:
    """Ordinary least squares (+ optional additive site terms) under 5-fold CV."""
    y = df["mag"].to_numpy(dtype=float)
    x_all = design_matrix(df, feature_cols)
    stations = df["station"].astype(str).to_numpy() if site else None
    networks = df["network"].astype(str).to_numpy() if site else None
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    preds = np.full(n, np.nan)

    for fi in range(n_folds):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        coef = _lstsq(x_all[tr], y[tr])
        base_tr = x_all[tr] @ coef
        base_te = x_all[te] @ coef
        if site:
            resid_tr = y[tr] - base_tr
            st_sums: dict[str, list[float]] = {}
            net_sums: dict[str, list[float]] = {}
            for s, net, r in zip(stations[tr], networks[tr], resid_tr):
                st_sums.setdefault(s, []).append(float(r))
                net_sums.setdefault(net, []).append(float(r))
            st_mean = {s: float(np.mean(v)) for s, v in st_sums.items() if len(v) >= min_station}
            net_mean = {s: float(np.mean(v)) for s, v in net_sums.items() if len(v) >= min_station}
            corr = np.array(
                [st_mean.get(stations[i], net_mean.get(networks[i], 0.0)) for i in te],
                dtype=float,
            )
            preds[te] = base_te + corr
        else:
            preds[te] = base_te

    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return {
        "r2": round(1.0 - ss_res / ss_tot, 3),
        "mae": round(float(np.abs(y - preds).mean()), 3),
        "rmse": round(float(np.sqrt(((y - preds) ** 2).mean())), 3),
        "n": int(n),
        "features": list(feature_cols),
        "predictions": preds,
        "residuals": y - preds,
    }


def cv_site_corrected(df: pd.DataFrame, *, min_station: int, seed: int) -> dict:
    """Backward-compatible Richter + site wrapper."""
    d = df.copy()
    d["log_d"] = np.log10(d["dist_km"].to_numpy(float) + 1.0)
    return cv_linear(
        d,
        ["log_peak_amp", "log_d"],
        seed=seed,
        site=True,
        min_station=min_station,
    )


def richter_only_cv(df: pd.DataFrame, seed: int = 0) -> dict:
    d = df.copy()
    d["log_d"] = np.log10(d["dist_km"].to_numpy(float) + 1.0)
    out = cv_linear(d, ["log_peak_amp", "log_d"], seed=seed, site=False)
    return {k: out[k] for k in ("r2", "mae", "rmse", "n")}


def assign_shape(row: pd.Series, *, peak_thr: float, coda_fast: float, coda_slow: float, onset_hi: float, onset_lo: float) -> str:
    peaks = float(row["n_rho_peaks"])
    coda = float(row["coda_slope"])
    onset = float(row["onset_sharp"])
    if peaks >= peak_thr:
        return "multipath"
    if onset >= onset_hi and coda <= coda_fast:
        return "impulsive_fastQ"
    if onset < onset_lo:
        return "emergent"
    if coda > coda_slow:
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
    if "src_region" not in d.columns and {"src_lat", "src_lon"}.issubset(d.columns):
        d["src_region"] = [
            f"{int(np.floor(a / 5) * 5):+d}/{int(np.floor(b / 5) * 5):+d}"
            if np.isfinite(a) and np.isfinite(b)
            else "unknown"
            for a, b in zip(d["src_lat"], d["src_lon"])
        ]

    # derived interpretable covariates
    d["log_d"] = np.log10(d["dist_km"].to_numpy(float) + 1.0)
    if "depth_km" in d.columns:
        d["log_depth"] = np.log10(np.clip(d["depth_km"].to_numpy(float), 0.0, None) + 1.0)
    else:
        d["log_depth"] = 0.0
    if "snr_db" in d.columns:
        d["snr_z"] = d["snr_db"].to_numpy(float)
        d["snr_z"] = np.nan_to_num(d["snr_z"], nan=float(np.nanmedian(d["snr_z"])))
    else:
        d["snr_z"] = 0.0

    ok = d["coda_slope"].notna() & d["dist_km"].notna()
    ld = d.loc[ok, "log_d"].to_numpy(float)
    cs = d.loc[ok, "coda_slope"].to_numpy(float)
    A = np.column_stack([ld, np.ones(len(ld))])
    ccoef = _lstsq(A, cs)
    d["coda_path_residual"] = np.nan
    d.loc[ok, "coda_path_residual"] = cs - A @ ccoef
    d["coda_path_residual"] = d["coda_path_residual"].fillna(0.0)

    # ---- magnitude ladder (all interpretable) ----
    feat_richter = ["log_peak_amp", "log_d"]
    feat_phys = ["log_peak_amp", "log_d", "log_depth", "snr_z"]
    feat_full = ["log_peak_amp", "log_d", "log_depth", "snr_z", "coda_path_residual"]

    mag_report: dict = {}
    r0 = cv_linear(d, feat_richter, seed=args.seed, site=False)
    mag_report["richter_all"] = {k: r0[k] for k in ("r2", "mae", "rmse", "n", "features")}
    r0s = cv_linear(d, feat_richter, seed=args.seed, site=True, min_station=args.min_station_events)
    mag_report["richter_site_all"] = {k: r0s[k] for k in ("r2", "mae", "rmse", "n", "features")}
    d["mag_pred_site"] = r0s["predictions"]
    d["mag_resid_site"] = r0s["residuals"]

    r_phys = cv_linear(d, feat_phys, seed=args.seed, site=True, min_station=args.min_station_events)
    mag_report["phys_site_all"] = {k: r_phys[k] for k in ("r2", "mae", "rmse", "n", "features")}
    r_full = cv_linear(d, feat_full, seed=args.seed, site=True, min_station=args.min_station_events)
    mag_report["phys_path_site_all"] = {k: r_full[k] for k in ("r2", "mae", "rmse", "n", "features")}

    # stratified ml/md with best interpretable feature set
    best_feats = feat_full
    pooled_pred = np.full(len(d), np.nan)
    covered = np.zeros(len(d), dtype=bool)
    per_type = {}
    for mt in ["ml", "md", "mb"]:
        sub_idx = np.where(d["mag_type"].to_numpy() == mt)[0]
        if len(sub_idx) < 40:
            continue
        sub = d.iloc[sub_idx].reset_index(drop=True)
        base = cv_linear(sub, feat_richter, seed=args.seed, site=False)
        site = cv_linear(sub, best_feats, seed=args.seed, site=True, min_station=max(2, args.min_station_events - 1))
        per_type[mt] = {
            "richter": {k: base[k] for k in ("r2", "mae", "rmse", "n")},
            "phys_path_site": {k: site[k] for k in ("r2", "mae", "rmse", "n", "features")},
        }
        pooled_pred[sub_idx] = site["predictions"]
        covered[sub_idx] = True
    if (~covered).any():
        pooled_pred[~covered] = r_full["predictions"][~covered]
    y = d["mag"].to_numpy(float)
    ss_res = float(((y - pooled_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    mag_report["per_type"] = per_type
    mag_report["stratified_phys_path_site"] = {
        "r2": round(1.0 - ss_res / ss_tot, 3),
        "mae": round(float(np.abs(y - pooled_pred).mean()), 3),
        "rmse": round(float(np.sqrt(((y - pooled_pred) ** 2).mean())), 3),
        "n": int(len(y)),
        "features": best_feats,
        "note": "ml/md/mb each get phys+path+site; rare types use global phys+path+site",
    }
    # keep old key name for README compatibility (points at best pooled model)
    mag_report["stratified_site_pooled"] = dict(mag_report["stratified_phys_path_site"])
    d["mag_pred_stratified"] = pooled_pred
    d["mag_resid_stratified"] = y - pooled_pred

    # export site tables from plain Richter residual (classic lookup)
    coef = richter_coef(d["log_peak_amp"].to_numpy(float), d["log_d"].to_numpy(float), y)
    base_all = apply_richter(coef, d["log_peak_amp"].to_numpy(float), d["log_d"].to_numpy(float))
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
        "form": "M ≈ a*log10(A) + b*log10(D+1) + c + site_term(station)  [+ depth/SNR/coda_path in extended models]",
        "a": round(float(coef[0]), 4),
        "b": round(float(coef[1]), 4),
        "c": round(float(coef[2]), 4),
        "n_site_terms": int(len(st_table)),
        "n_network_terms": int(len(net_table)),
        "extended_features": best_feats,
    }

    # ---- 2-D taxonomy: data-driven shape thresholds ----
    peak_thr = float(d["n_rho_peaks"].quantile(0.67))
    coda_fast = float(d["coda_slope"].quantile(0.33))  # more negative
    coda_slow = float(d["coda_slope"].quantile(0.67))
    onset_hi = float(d["onset_sharp"].quantile(0.67))
    onset_lo = float(d["onset_sharp"].quantile(0.33))
    shape_thr = {
        "peak_thr": round(peak_thr, 3),
        "coda_fast": round(coda_fast, 3),
        "coda_slow": round(coda_slow, 3),
        "onset_hi": round(onset_hi, 3),
        "onset_lo": round(onset_lo, 3),
    }
    q33, q66 = float(d["reduced_amp"].quantile(0.33)), float(d["reduced_amp"].quantile(0.66))
    d["shape"] = d.apply(
        lambda r: assign_shape(
            r, peak_thr=peak_thr, coda_fast=coda_fast, coda_slow=coda_slow,
            onset_hi=onset_hi, onset_lo=onset_lo,
        ),
        axis=1,
    )
    d["strength"] = [assign_strength(v, q33, q66) for v in d["reduced_amp"]]
    d["tax2d"] = d["shape"] + "×" + d["strength"]

    disc = {
        "shape_thresholds": shape_thr,
        "shape_vs_region": region_association(d, "shape", "path_region"),
        "shape_vs_src_region": region_association(d, "shape", "src_region") if "src_region" in d.columns else {},
        "strength_vs_region": region_association(d, "strength", "path_region"),
        "tax2d_vs_region": region_association(d, "tax2d", "path_region"),
        "shape_counts": d["shape"].value_counts().to_dict(),
        "strength_counts": d["strength"].value_counts().to_dict(),
        "tax2d_counts": d["tax2d"].value_counts().to_dict(),
        "reduced_amp_terciles": {"q33": round(q33, 3), "q66": round(q66, 3)},
    }

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
            "top_path_regions": d.loc[d["shape"] == sh, "path_region"].value_counts().head(3).to_dict(),
            "top_src_regions": (
                d.loc[d["shape"] == sh, "src_region"].value_counts().head(3).to_dict()
                if "src_region" in d.columns else {}
            ),
        })
    disc["shape_path_effects"] = shape_effects

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

    best_r2 = mag_report["stratified_phys_path_site"]["r2"]
    report = {
        "n_traces": int(len(d)),
        "magnitude": mag_report,
        "discrimination": disc,
        "headline": {
            "richter_r2": mag_report["richter_all"]["r2"],
            "site_r2": mag_report["richter_site_all"]["r2"],
            "phys_site_r2": mag_report["phys_site_all"]["r2"],
            "phys_path_site_r2": mag_report["phys_path_site_all"]["r2"],
            "stratified_site_r2": best_r2,
            "stratified_site_mae": mag_report["stratified_phys_path_site"]["mae"],
            "shape_region_V": disc["shape_vs_region"]["cramers_v"],
            "shape_region_V_0_50km": disc["shape_vs_region"]["distance_controlled"].get("0-50", {}).get("cramers_v"),
            "shape_src_region_V": disc.get("shape_vs_src_region", {}).get("cramers_v"),
            "tax2d_region_V": disc["tax2d_vs_region"]["cramers_v"],
            "ceiling_note": (
                "Interpretable single-station ceiling on STEAD is ~0.83–0.88 R² with site+depth+SNR+path. "
                "0.95 is not realistic without multi-station / unified magnitude scale."
            ),
        },
    }
    (out / "ceiling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    d.to_csv(out / "traces_labeled.csv", index=False)

    # human-readable interpretability note
    md = [
        "# Interpretable physics ceiling",
        "",
        f"- traces: **{len(d)}**",
        f"- Richter R²: **{mag_report['richter_all']['r2']}**",
        f"- + site terms R²: **{mag_report['richter_site_all']['r2']}**",
        f"- + depth/SNR + site R²: **{mag_report['phys_site_all']['r2']}**",
        f"- + coda path residual + site R²: **{mag_report['phys_path_site_all']['r2']}**",
        f"- **best (ml/md stratified phys+path+site) R²: {best_r2}**  MAE={mag_report['stratified_phys_path_site']['mae']}",
        "",
        "## Taxonomy `shape × strength`",
        f"- shape↔path-region Cramér's V: **{disc['shape_vs_region']['cramers_v']}**",
        f"- (0–50 km controlled): **{disc['shape_vs_region']['distance_controlled'].get('0-50', {}).get('cramers_v')}**",
        f"- tax2d↔path-region V: **{disc['tax2d_vs_region']['cramers_v']}**",
        f"- thresholds: `{json.dumps(shape_thr)}`",
        "",
        "## Shape → structure (coda path residual)",
    ]
    for e in shape_effects:
        md.append(
            f"- **{e['shape']}** n={e['n']}: residual={e['coda_residual_mean']}, "
            f"d={e['cohens_d_vs_rest']}, maḡ={e['mag_mean']}, regions={e['top_path_regions']}"
        )
    md.append("")
    md.append("## Reading")
    md.append(
        "- **strength** (reduced_amp) tracks source size → magnitude.\n"
        "- **shape** (onset/coda/ρ peaks) tracks path/mechanism → geography & Q-like residual.\n"
        "- Do not chase R²→0.95 with single-station interpretable features."
    )
    (out / "INTERPRETABILITY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax = axes[0, 0]
        ax.scatter(d["mag"], d["mag_pred_stratified"], s=12, alpha=0.45, c="steelblue")
        lims = [0, max(6, float(d["mag"].max()))]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("catalog mag")
        ax.set_ylabel("predicted mag")
        ax.set_title(f"Best interpretable  R²={best_r2}  MAE={mag_report['stratified_phys_path_site']['mae']}")

        ax = axes[0, 1]
        order = ["weak", "mid", "strong"]
        data = [d.loc[d["strength"] == s, "mag"].to_numpy() for s in order]
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_ylabel("magnitude")
        ax.set_title("Strength axis (reduced_amp terciles)")

        ax = axes[1, 0]
        shapes = sorted(d["shape"].unique())
        data = [d.loc[d["shape"] == s, "mag"].to_numpy() for s in shapes]
        ax.boxplot(data, labels=shapes, showfliers=False)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_ylabel("magnitude")
        ax.set_title(f"Shape axis (path V={disc['shape_vs_region']['cramers_v']})")

        ax = axes[1, 1]
        data = [d.loc[d["shape"] == s, "coda_path_residual"].dropna().to_numpy() for s in shapes]
        ax.boxplot(data, labels=shapes, showfliers=False)
        ax.axhline(0, color="k", lw=0.6)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_ylabel("coda path residual")
        ax.set_title("Structure proxy by shape")

        fig.tight_layout()
        fig.savefig(out / "ceiling_overview.png", dpi=130)
        print(f"[ceiling] wrote {out / 'ceiling_overview.png'}", flush=True)
    except Exception as exc:
        print(f"[ceiling] plot skipped: {exc}", flush=True)

    print(json.dumps(report["headline"], indent=2), flush=True)


if __name__ == "__main__":
    main()
