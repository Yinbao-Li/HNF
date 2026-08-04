#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geographic morphology anomalies: dominant regional shape vs rare outsiders.

Question
--------
In a region where one of the five causal shapes dominates, do the rare
mismatched events look like **path / structure outliers** (extreme
distance-detrended coda residual), or like source / distance confounders?

CPU-only; does not touch GPU training.

Example
-------
  PYTHONPATH=. python tools/analyze_geo_morphology_anomalies.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


SHAPE_ORDER = (
    "impulsive_fastQ",
    "emergent",
    "multipath",
    "slow_coda",
    "standard",
)

SHAPE_COLORS = {
    "impulsive_fastQ": "#0072B2",
    "emergent": "#E69F00",
    "multipath": "#009E73",
    "slow_coda": "#D55E00",
    "standard": "#7F7F7F",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geographic morphology anomaly analysis")
    p.add_argument(
        "--traces",
        default="outputs/interpretable_physics_best/ceiling/traces_labeled.csv",
    )
    p.add_argument("--min-region-n", type=int, default=40)
    p.add_argument("--min-dominant-frac", type=float, default=0.45)
    p.add_argument("--max-anomaly-frac", type=float, default=0.20)
    p.add_argument("--coastline", default="docs/figures/geo/ne_110m_land.geojson")
    p.add_argument("--output-dir", default="outputs/geo_morphology_anomalies")
    p.add_argument("--fig-dir", default="docs/figures")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"diff": float("nan"), "ci95": [float("nan"), float("nan")], "p_two_sided": float("nan")}
    obs = float(a.mean() - b.mean())
    diffs = []
    for _ in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(aa.mean() - bb.mean()))
    diffs = np.asarray(diffs)
    # two-sided p vs 0 under bootstrap distribution shifted to null
    null = diffs - diffs.mean()
    p = float(np.mean(np.abs(null) >= abs(obs)))
    return {
        "diff": obs,
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_two_sided": p,
    }


def find_regional_anomalies(df: pd.DataFrame, min_n: int, min_dom: float, max_anom: float) -> pd.DataFrame:
    rows = []
    for region, sub in df.groupby("path_region"):
        if len(sub) < min_n:
            continue
        vc = sub["shape"].value_counts()
        dom_shape = str(vc.index[0])
        dom_n = int(vc.iloc[0])
        dom_frac = dom_n / len(sub)
        if dom_frac < min_dom:
            continue
        # anomalies = everything that is not the dominant shape and is rare enough
        for sh, n_sh in vc.items():
            sh = str(sh)
            if sh == dom_shape:
                continue
            frac = n_sh / len(sub)
            if frac > max_anom:
                continue
            if n_sh < 2:
                continue
            rows.append(
                {
                    "path_region": region,
                    "n_region": int(len(sub)),
                    "dominant_shape": dom_shape,
                    "dominant_n": dom_n,
                    "dominant_frac": round(dom_frac, 3),
                    "anomaly_shape": sh,
                    "anomaly_n": int(n_sh),
                    "anomaly_frac": round(frac, 3),
                }
            )
    return pd.DataFrame(rows)


def compare_within_region(df: pd.DataFrame, region: str, dom: str, anom: str) -> dict:
    sub = df[df["path_region"] == region]
    a = sub[sub["shape"] == anom]
    d = sub[sub["shape"] == dom]
    # distance-matched peers: dominant events with dist in anomaly IQR
    if len(a) >= 2 and a["dist_km"].notna().any():
        lo, hi = a["dist_km"].quantile(0.1), a["dist_km"].quantile(0.9)
        d_match = d[(d["dist_km"] >= lo) & (d["dist_km"] <= hi)]
        if len(d_match) < 5:
            d_match = d
    else:
        d_match = d

    metrics = [
        "coda_path_residual",
        "coda_slope",
        "onset_sharp",
        "n_rho_peaks",
        "dist_km",
        "depth_km",
        "mag",
        "snr_db",
        "reduced_amp",
    ]
    out = {
        "path_region": region,
        "dominant_shape": dom,
        "anomaly_shape": anom,
        "n_anomaly": int(len(a)),
        "n_dominant": int(len(d)),
        "n_dominant_dist_matched": int(len(d_match)),
    }
    for m in metrics:
        if m not in a.columns:
            continue
        aa = a[m].to_numpy(float)
        dd = d_match[m].to_numpy(float)
        out[f"{m}_anom_mean"] = float(np.nanmean(aa)) if np.isfinite(aa).any() else float("nan")
        out[f"{m}_dom_mean"] = float(np.nanmean(dd)) if np.isfinite(dd).any() else float("nan")
        out[f"{m}_cohen_d"] = _cohen_d(aa, dd)
        boot = _bootstrap_mean_diff(aa, dd)
        out[f"{m}_diff"] = boot["diff"]
        out[f"{m}_diff_ci95"] = boot["ci95"]
        out[f"{m}_p"] = boot["p_two_sided"]

    # Structure-leaning score: extreme |coda residual| relative to local dominant,
    # while |dist| and |mag| differences are small.
    c_d = abs(out.get("coda_path_residual_cohen_d", float("nan")))
    dist_d = abs(out.get("dist_km_cohen_d", float("nan")))
    mag_d = abs(out.get("mag_cohen_d", float("nan")))
    out["structure_leaning"] = bool(
        np.isfinite(c_d)
        and c_d >= 0.5
        and (not np.isfinite(dist_d) or dist_d < 0.6)
        and (not np.isfinite(mag_d) or mag_d < 0.6)
    )
    # Interpretation
    if out["structure_leaning"]:
        sign = out.get("coda_path_residual_diff", 0.0)
        path_note = (
            "more negative residual → faster coda / lower-Q or stronger attenuation than local peers"
            if sign < 0
            else "more positive residual → slower coda / higher-Q or scattering-richer than local peers"
        )
        out["interpretation"] = (
            f"Likely **path/structure candidate**: anomaly shape '{anom}' in "
            f"{dom}-dominated {region}; coda residual differs from distance-matched locals "
            f"(Cohen d={out.get('coda_path_residual_cohen_d', float('nan')):+.2f}) while "
            f"distance/magnitude are similar. {path_note}"
        )
    else:
        reasons = []
        if np.isfinite(dist_d) and dist_d >= 0.6:
            reasons.append("distance mismatch")
        if np.isfinite(mag_d) and mag_d >= 0.6:
            reasons.append("magnitude/source mismatch")
        if not np.isfinite(c_d) or c_d < 0.5:
            reasons.append("coda residual not extreme vs locals")
        out["interpretation"] = (
            f"Ambiguous / likely confounder: anomaly '{anom}' in {region} — "
            + (", ".join(reasons) if reasons else "insufficient evidence for path anomaly")
            + ". Could be source radiation, site, SNR, or taxonomy noise."
        )
    return out


def load_coast(path: Path) -> list[np.ndarray]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    polys: list[np.ndarray] = []

    def walk(coords):
        if not coords:
            return
        if isinstance(coords[0][0], (int, float)):
            arr = np.asarray(coords, float)
            if arr.ndim == 2 and len(arr) >= 3:
                polys.append(arr)
            return
        for c in coords:
            walk(c)

    for feat in data.get("features", []):
        walk((feat.get("geometry") or {}).get("coordinates") or [])
    return polys


def make_figure(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    comps: list[dict],
    coast: list[np.ndarray],
    out_png: Path,
    dpi: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Build anomaly event table
    anom_mask = np.zeros(len(df), dtype=bool)
    for _, r in pairs.iterrows():
        m = (df["path_region"] == r["path_region"]) & (df["shape"] == r["anomaly_shape"])
        anom_mask |= m.to_numpy()
    anom_df = df.loc[anom_mask].copy()

    fig = plt.figure(figsize=(7.4, 8.2), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.28, left=0.09, right=0.98, top=0.94, bottom=0.07)

    # (a) map: dominant background + anomaly stars
    ax = fig.add_subplot(gs[0, :])
    ax.text(-0.03, 1.06, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.set_title("Regional dominant morphology vs rare mismatched events", loc="left", pad=4)
    for ring in coast:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#ECE9E1", edgecolor="#B7B2A6", lw=0.25, zorder=0)
    # plot all events lightly by shape
    for sh in SHAPE_ORDER:
        sub = df[df["shape"] == sh]
        ax.scatter(sub["src_lon"], sub["src_lat"], s=6, c=SHAPE_COLORS[sh], alpha=0.25, lw=0, rasterized=True, zorder=2)
    # anomalies emphasized
    for sh in SHAPE_ORDER:
        sub = anom_df[anom_df["shape"] == sh]
        if sub.empty:
            continue
        ax.scatter(
            sub["src_lon"],
            sub["src_lat"],
            s=36,
            c=SHAPE_COLORS[sh],
            marker="*",
            edgecolors="k",
            linewidths=0.35,
            alpha=0.95,
            zorder=5,
            label=f"anomaly {sh} (n={len(sub)})",
        )
    # circle enriched regions with many anomalies
    for region in pairs["path_region"].unique():
        # parse +35/-125
        try:
            lat_s, lon_s = str(region).split("/")
            lat0 = float(lat_s)
            lon0 = float(lon_s)
        except Exception:
            continue
        ax.add_patch(Circle((lon0, lat0), 3.2, fill=False, ec="#444444", lw=0.7, ls=":", zorder=4))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-50, 75)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D9D4C8", lw=0.4)
    ax.legend(loc="lower left", fontsize=6.5, frameon=True, fancybox=False, edgecolor="#D0CBC0", framealpha=0.92)

    # (b) Cohen d forest for structure vs confounders
    axb = fig.add_subplot(gs[1, 0])
    axb.text(-0.12, 1.06, "(b)", transform=axb.transAxes, fontsize=11, fontweight="bold", va="top")
    axb.set_title("Anomaly vs distance-matched locals (Cohen's d)", loc="left", pad=3)
    metrics = ["coda_path_residual", "dist_km", "mag", "onset_sharp", "n_rho_peaks", "snr_db"]
    labels = ["coda residual\n(path/Q)", "distance", "magnitude", "onset sharp", "ρ multipath", "SNR"]
    # aggregate across structure-leaning comps
    lean = [c for c in comps if c.get("structure_leaning")]
    pool = lean if lean else comps
    ys = np.arange(len(metrics))
    means, los, his = [], [], []
    for m in metrics:
        vals = [c.get(f"{m}_cohen_d", float("nan")) for c in pool]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            means.append(0.0)
            los.append(0.0)
            his.append(0.0)
        else:
            means.append(float(np.mean(vals)))
            los.append(float(np.percentile(vals, 25)))
            his.append(float(np.percentile(vals, 75)))
    axb.axvline(0, color="#999999", lw=0.7, ls="--")
    axb.axvspan(-0.5, 0.5, color="#EEEAE2", alpha=0.7, zorder=0)
    for y, mu, lo, hi in zip(ys, means, los, his):
        col = "#C44E52" if abs(mu) >= 0.5 and metrics[int(y)] == "coda_path_residual" else "#4C72B0"
        axb.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round")
        axb.scatter([mu], [y], color=col, s=28, zorder=3)
    axb.set_yticks(ys)
    axb.set_yticklabels(labels)
    axb.set_xlabel("Cohen's d (anomaly − local dominant)")
    axb.set_xlim(-2.2, 2.2)
    axb.text(0.02, 0.02, f"avg over {len(pool)} region×anomaly pairs", transform=axb.transAxes, fontsize=6.5, color="#666")

    # (c) coda residual distributions: anomaly vs dominant in top regions
    axc = fig.add_subplot(gs[1, 1])
    axc.text(-0.12, 1.06, "(c)", transform=axc.transAxes, fontsize=11, fontweight="bold", va="top")
    axc.set_title("Coda path residual: local dominant vs anomalies", loc="left", pad=3)
    # pick up to 4 richest anomaly regions
    top_regs = (
        pairs.groupby("path_region")["anomaly_n"].sum().sort_values(ascending=False).head(4).index.tolist()
    )
    xpos = 0
    xticks, xlabels = [], []
    for reg in top_regs:
        sub = df[df["path_region"] == reg]
        dom = pairs.loc[pairs["path_region"] == reg, "dominant_shape"].iloc[0]
        dvals = sub.loc[sub["shape"] == dom, "coda_path_residual"].dropna().to_numpy()
        avals = sub.loc[sub["shape"] != dom, "coda_path_residual"].dropna().to_numpy()
        # only keep shapes that appear as anomaly pairs for this region
        anom_shapes = set(pairs.loc[pairs["path_region"] == reg, "anomaly_shape"])
        avals = sub.loc[sub["shape"].isin(anom_shapes), "coda_path_residual"].dropna().to_numpy()
        parts = axc.violinplot([dvals, avals], positions=[xpos, xpos + 0.55], widths=0.45, showextrema=False, showmeans=False)
        for body, col in zip(parts["bodies"], ["#7F7F7F", "#C44E52"]):
            body.set_facecolor(col)
            body.set_alpha(0.35)
            body.set_edgecolor(col)
        axc.scatter([xpos], [np.nanmedian(dvals)], color="#7F7F7F", s=16, zorder=3)
        axc.scatter([xpos + 0.55], [np.nanmedian(avals) if len(avals) else np.nan], color="#C44E52", s=16, zorder=3)
        xticks.append(xpos + 0.27)
        xlabels.append(f"{reg}\n{dom}")
        xpos += 1.5
    axc.axhline(0, color="#999", lw=0.7, ls="--")
    axc.set_xticks(xticks)
    axc.set_xticklabels(xlabels, fontsize=7)
    axc.set_ylabel("Coda path residual")
    handles = [
        Line2D([0], [0], color="#7F7F7F", lw=6, alpha=0.5, label="regional dominant"),
        Line2D([0], [0], color="#C44E52", lw=6, alpha=0.5, label="regional anomalies"),
    ]
    axc.legend(handles=handles, loc="upper right", fontsize=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.traces)
    need = {"shape", "path_region", "src_lat", "src_lon", "coda_path_residual", "dist_km"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    df = df[df["shape"].isin(SHAPE_ORDER)].copy()

    pairs = find_regional_anomalies(df, args.min_region_n, args.min_dominant_frac, args.max_anomaly_frac)
    print(f"[geo-anom] regions with dominant+rare pattern: {len(pairs)} pairs", flush=True)

    comps = []
    for _, r in pairs.iterrows():
        comps.append(compare_within_region(df, r["path_region"], r["dominant_shape"], r["anomaly_shape"]))

    lean = [c for c in comps if c.get("structure_leaning")]
    # Global pooled test: all anomaly events in dominant regions vs their local dominants
    anom_resids, dom_resids = [], []
    for _, r in pairs.iterrows():
        sub = df[df["path_region"] == r["path_region"]]
        a = sub.loc[sub["shape"] == r["anomaly_shape"], "coda_path_residual"].to_numpy(float)
        d = sub.loc[sub["shape"] == r["dominant_shape"], "coda_path_residual"].to_numpy(float)
        anom_resids.append(a)
        dom_resids.append(d)
    anom_all = np.concatenate(anom_resids) if anom_resids else np.array([])
    dom_all = np.concatenate(dom_resids) if dom_resids else np.array([])
    global_d = _cohen_d(anom_all, dom_all)
    global_boot = _bootstrap_mean_diff(anom_all, dom_all)

    # Also: |residual| larger for anomalies?
    abs_d = _cohen_d(np.abs(anom_all), np.abs(dom_all))

    report = {
        "n_traces": int(len(df)),
        "settings": {
            "min_region_n": args.min_region_n,
            "min_dominant_frac": args.min_dominant_frac,
            "max_anomaly_frac": args.max_anomaly_frac,
        },
        "n_region_anomaly_pairs": int(len(pairs)),
        "n_structure_leaning_pairs": int(len(lean)),
        "global_anomaly_vs_dominant_coda_residual": {
            "n_anomaly": int(np.isfinite(anom_all).sum()),
            "n_dominant": int(np.isfinite(dom_all).sum()),
            "cohen_d": global_d,
            "mean_diff": global_boot["diff"],
            "ci95": global_boot["ci95"],
            "p_two_sided": global_boot["p_two_sided"],
            "cohen_d_abs_residual": abs_d,
        },
        "pairs": pairs.to_dict(orient="records"),
        "comparisons": comps,
        "headline_reading": (
            "Rare mismatched morphologies inside a shape-dominated region are a **useful "
            "screening list** for path/structure candidates, but they are **not** automatically "
            "underground anomalies. Prefer cases where distance-detrended coda residual differs "
            "from distance-matched local peers while magnitude/distance do not."
        ),
        "structure_leaning_examples": [
            {
                "path_region": c["path_region"],
                "dominant_shape": c["dominant_shape"],
                "anomaly_shape": c["anomaly_shape"],
                "n_anomaly": c["n_anomaly"],
                "coda_path_residual_cohen_d": c.get("coda_path_residual_cohen_d"),
                "interpretation": c["interpretation"],
            }
            for c in lean
        ],
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "region_anomaly_pairs.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "anomaly_vs_dominant_stats.csv", index=False)
    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown report
    lines = [
        "# Geographic morphology anomalies",
        "",
        report["headline_reading"],
        "",
        f"- Region×anomaly pairs: **{len(pairs)}**",
        f"- Structure-leaning pairs: **{len(lean)}** / {len(comps)}",
        f"- Global coda residual Cohen's d (anomaly − dominant): **{global_d:+.3f}** "
        f"(95% CI {global_boot['ci95'][0]:+.3f}..{global_boot['ci95'][1]:+.3f}, "
        f"bootstrap p≈{global_boot['p_two_sided']:.3f})",
        f"- Cohen's d on |residual|: **{abs_d:+.3f}**",
        "",
        "## Structure-leaning examples",
        "",
    ]
    if lean:
        for c in lean[:12]:
            lines.append(
                f"- `{c['path_region']}` dominant=`{c['dominant_shape']}` anomaly=`{c['anomaly_shape']}` "
                f"(n={c['n_anomaly']}): coda d={c.get('coda_path_residual_cohen_d', float('nan')):+.2f}"
            )
            lines.append(f"  - {c['interpretation']}")
    else:
        lines.append("- None passed the structure-leaning filter on this slice.")
    lines += [
        "",
        "## Caveats",
        "",
        "1. Shape labels are waveform morphology, not tomography.",
        "2. STEAD sampling is network-biased (ZQ / western US heavy).",
        "3. Source radiation pattern and site response can mimic path effects.",
        "4. Treat structure-leaning anomalies as **hypotheses** for follow-up "
        "(local geology / Vs30 / tomography / multi-station stacks).",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    coast = load_coast(Path(args.coastline))
    fig_dir = Path(args.fig_dir)
    fig_path = fig_dir / "geo_morphology_anomalies.png"
    make_figure(df, pairs, comps, coast, fig_path, args.dpi)

    print(f"[geo-anom] → {out / 'REPORT.md'}", flush=True)
    print(f"[geo-anom] → {fig_path}", flush=True)
    print(
        f"[geo-anom] structure-leaning={len(lean)}/{len(comps)}  "
        f"global coda d={global_d:+.3f}  |resid| d={abs_d:+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
