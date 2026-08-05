#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regional temporal evolution of 5 morphology classes + coda residual vs seismicity.

Uses:
  - labeled traces (shape, coda_path_residual, lat/lon) for waveform–attenuation state
  - STEAD chunk CSVs (deduped by source_id) for regional activity / M≥X markers

Example
-------
  PYTHONPATH=. python tools/analyze_shape_temporal_evolution.py \\
    --traces outputs/interpretable_physics_best/ceiling/traces_labeled.csv \\
    --regions socal,pnw,alaska --freq Q \\
    --output-dir outputs/shape_temporal_evolution
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEAD_DIR

SHAPE_ORDER = ["impulsive_fastQ", "emergent", "multipath", "slow_coda", "standard"]
SHAPE_COLORS = {
    "impulsive_fastQ": "#2F6FED",
    "emergent": "#D4A017",
    "multipath": "#2E8B57",
    "slow_coda": "#B85C38",
    "standard": "#6B7280",
}
C_INK = "#1C1917"
C_MUTED = "#78716C"
C_LINE = "#D6D3D1"
BG = "#FFFEF9"

# Named boxes aligned with the shape–geo–coda enrichment narrative.
REGIONS = {
    "socal": {
        "title": "Southern California",
        "lat": (32.0, 36.5),
        "lon": (-122.0, -114.0),
        "note": "dense STEAD coverage; path_region ~+30/−120, +35/−120",
    },
    "pnw": {
        "title": "Pacific Northwest / Cascadia arc",
        "lat": (40.0, 50.5),
        "lon": (-130.0, -120.0),
        "note": "impulsive_fastQ enrichment narrative (W.US / NE Pac.)",
    },
    "cascadia_volc": {
        "title": "Cascadia volcanic arc / St Helens–Rainier–Puget",
        "lat": (45.2, 48.8),
        "lon": (-124.6, -120.2),
        "note": "dense local sampling for PNW same-station replication of Salton-like β residuals",
    },
    "alaska": {
        "title": "Southern Alaska / Aleutians east",
        "lat": (55.0, 65.0),
        "lon": (-165.0, -145.0),
        "note": "high STEAD density; path_region ~+60/−155",
    },
    "camerica": {
        "title": "Central America / E. Pacific",
        "lat": (5.0, 25.0),
        "lon": (-110.0, -75.0),
        "note": "slow_coda enrichment narrative",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shape / β_res temporal evolution vs seismicity")
    p.add_argument(
        "--traces",
        default="outputs/interpretable_physics_best/ceiling/traces_labeled.csv",
    )
    p.add_argument("--stead-dir", default=str(STEAD_DIR))
    p.add_argument("--regions", default="socal,pnw,alaska")
    p.add_argument("--freq", default="Q", choices=["M", "Q", "Y"], help="time bin: month/quarter/year")
    p.add_argument("--mag-mod", type=float, default=3.0, help="moderate event threshold")
    p.add_argument("--mag-strong", type=float, default=5.0, help="strong event threshold")
    p.add_argument("--pre-days", type=float, default=90.0)
    p.add_argument("--post-days", type=float, default=90.0)
    p.add_argument("--min-bin-n", type=int, default=5, help="min labeled traces per bin for fractions")
    p.add_argument("--output-dir", default="outputs/shape_temporal_evolution")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--fixed-only",
        action="store_true",
        help="If traces have fixed_station column, keep only multi-year fixed stations",
    )
    return p.parse_args()


def _parse_trace_time(name: str) -> pd.Timestamp:
    m = re.search(r"_(\d{14})_", str(name))
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", errors="coerce")


def load_labeled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"trace_name", "shape", "coda_path_residual", "src_lat", "src_lon", "mag"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"labeled CSV missing columns: {sorted(miss)}")
    df = df[df["shape"].isin(SHAPE_ORDER)].copy()
    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
    else:
        df["t"] = pd.NaT
    miss_t = df["t"].isna()
    if miss_t.any():
        df.loc[miss_t, "t"] = df.loc[miss_t, "trace_name"].map(_parse_trace_time)
    df = df.dropna(subset=["t", "src_lat", "src_lon"]).sort_values("t")
    df["beta"] = pd.to_numeric(df["coda_path_residual"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    if "fixed_station" in df.columns:
        df["fixed_station"] = df["fixed_station"].astype(str).str.lower().isin(["1", "true", "yes"])
    return df


def load_stead_events_union(stead_dir: Path, regions: list[str]) -> pd.DataFrame:
    """Load STEAD events once for the union of requested region boxes, then filter per region."""
    usecols = [
        "source_id",
        "source_origin_time",
        "source_latitude",
        "source_longitude",
        "source_magnitude",
        "trace_category",
    ]
    lat_min = min(REGIONS[k]["lat"][0] for k in regions)
    lat_max = max(REGIONS[k]["lat"][1] for k in regions)
    lon_min = min(REGIONS[k]["lon"][0] for k in regions)
    lon_max = max(REGIONS[k]["lon"][1] for k in regions)
    frames = []
    for csv in sorted(Path(stead_dir).glob("chunk*_eofextract/chunk*.csv")):
        print(f"[stead] reading {csv.name}", flush=True)
        df = pd.read_csv(csv, usecols=usecols, low_memory=False)
        if "trace_category" in df.columns:
            df = df[df["trace_category"].astype(str).str.contains("earthquake", case=False, na=False)]
        lat = pd.to_numeric(df["source_latitude"], errors="coerce")
        lon = pd.to_numeric(df["source_longitude"], errors="coerce")
        mag = pd.to_numeric(df["source_magnitude"], errors="coerce")
        # crude lon box (may be wide for multi-region); fine for one-pass load
        m = lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max) & mag.notna()
        if not m.any():
            continue
        sub = df.loc[m, ["source_id", "source_origin_time", "source_latitude", "source_longitude", "source_magnitude"]].copy()
        sub["source_origin_time"] = pd.to_datetime(sub["source_origin_time"], errors="coerce")
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["source_id", "t", "lat", "lon", "mag"])
    cat = pd.concat(frames, ignore_index=True).dropna(subset=["source_origin_time", "source_id"])
    cat = (
        cat.sort_values(["source_id", "source_magnitude"], ascending=[True, False])
        .groupby("source_id", as_index=False)
        .first()
    )
    return cat.rename(
        columns={
            "source_origin_time": "t",
            "source_latitude": "lat",
            "source_longitude": "lon",
            "source_magnitude": "mag",
        }
    ).sort_values("t")


def filter_box(cat: pd.DataFrame, lat_rng, lon_rng) -> pd.DataFrame:
    if cat.empty:
        return cat
    m = cat["lat"].between(*lat_rng) & cat["lon"].between(*lon_rng)
    return cat.loc[m].copy()


def _period_index(t: pd.Series, freq: str) -> pd.PeriodIndex:
    return t.dt.to_period(freq)


def binned_state(df: pd.DataFrame, freq: str, min_bin_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.copy()
    g["period"] = _period_index(g["t"], freq)
    rows = []
    for per, sub in g.groupby("period"):
        n = len(sub)
        fr = {sh: float((sub["shape"] == sh).mean()) for sh in SHAPE_ORDER}
        row = {
            "period": str(per),
            "t_mid": per.to_timestamp(how="start") + (per.end_time - per.start_time) / 2,
            "n_labeled": n,
            "beta_median": float(sub["beta"].median()) if sub["beta"].notna().any() else float("nan"),
            "beta_mean": float(sub["beta"].mean()) if sub["beta"].notna().any() else float("nan"),
            "frac_impulsive": fr["impulsive_fastQ"],
            "frac_slow_coda": fr["slow_coda"],
            "frac_multipath": fr["multipath"],
            "frac_emergent": fr["emergent"],
            "frac_standard": fr["standard"],
        }
        for sh in SHAPE_ORDER:
            row[f"n_{sh}"] = int((sub["shape"] == sh).sum())
            row[f"frac_{sh}"] = fr[sh] if n >= min_bin_n else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("t_mid")
    return out


def binned_activity(cat: pd.DataFrame, freq: str, mag_mod: float, mag_strong: float) -> pd.DataFrame:
    if cat.empty:
        return pd.DataFrame()
    g = cat.copy()
    g["period"] = _period_index(g["t"], freq)
    rows = []
    for per, sub in g.groupby("period"):
        rows.append(
            {
                "period": str(per),
                "t_mid": per.to_timestamp(how="start") + (per.end_time - per.start_time) / 2,
                "n_events": int(len(sub)),
                "n_mod": int((sub["mag"] >= mag_mod).sum()),
                "n_strong": int((sub["mag"] >= mag_strong).sum()),
                "max_mag": float(sub["mag"].max()),
                "mean_mag": float(sub["mag"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("t_mid")


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(m.sum())
    if n < 6:
        return {"n": n, "rho": float("nan"), "p": float("nan")}
    # rank correlation without scipy dependency
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # two-sided t approx
    if abs(rho) >= 1:
        p = 0.0
    else:
        tstat = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho**2))
        # regularized incomplete-beta approx via erfc for df large; simple normal fallback
        from math import erfc, sqrt

        p = float(erfc(abs(tstat) / sqrt(2.0)))  # rough two-sided normal approx
    return {"n": n, "rho": rho, "p": p}


def event_window_tests(
    labeled: pd.DataFrame,
    strong_events: pd.DataFrame,
    pre_days: float,
    post_days: float,
) -> list[dict]:
    """Compare β_res / shape mix in pre vs post windows around each strong event."""
    out = []
    if labeled.empty or strong_events.empty:
        return out
    pre = pd.Timedelta(days=pre_days)
    post = pd.Timedelta(days=post_days)
    gap = pd.Timedelta(days=7)  # exclude ±1 week around origin
    for _, ev in strong_events.iterrows():
        t0 = ev["t"]
        pre_m = (labeled["t"] >= t0 - pre) & (labeled["t"] < t0 - gap)
        post_m = (labeled["t"] > t0 + gap) & (labeled["t"] <= t0 + post)
        a = labeled.loc[pre_m]
        b = labeled.loc[post_m]
        if len(a) < 3 or len(b) < 3:
            continue
        rec = {
            "source_id": str(ev["source_id"]),
            "t": t0.isoformat(),
            "mag": float(ev["mag"]),
            "n_pre": int(len(a)),
            "n_post": int(len(b)),
            "beta_pre": float(a["beta"].median()),
            "beta_post": float(b["beta"].median()),
            "delta_beta": float(b["beta"].median() - a["beta"].median()),
            "frac_impulsive_pre": float((a["shape"] == "impulsive_fastQ").mean()),
            "frac_impulsive_post": float((b["shape"] == "impulsive_fastQ").mean()),
            "frac_slow_pre": float((a["shape"] == "slow_coda").mean()),
            "frac_slow_post": float((b["shape"] == "slow_coda").mean()),
        }
        # Mann–Whitney via rank-sum normal approx on beta
        xa = a["beta"].dropna().to_numpy()
        xb = b["beta"].dropna().to_numpy()
        if len(xa) >= 3 and len(xb) >= 3:
            allv = np.concatenate([xa, xb])
            ranks = pd.Series(allv).rank().to_numpy()
            ra = ranks[: len(xa)].sum()
            na, nb = len(xa), len(xb)
            u = ra - na * (na + 1) / 2.0
            mu = na * nb / 2.0
            sig = np.sqrt(na * nb * (na + nb + 1) / 12.0)
            z = (u - mu) / max(sig, 1e-9)
            from math import erfc, sqrt

            rec["beta_mw_p"] = float(erfc(abs(z) / sqrt(2.0)))
            rec["beta_mw_z"] = float(z)
        out.append(rec)
    return out


def plot_region(
    region_key: str,
    meta: dict,
    state: pd.DataFrame,
    act: pd.DataFrame,
    strong: pd.DataFrame,
    window_tests: list[dict],
    corr: dict,
    out_png: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(11.0, 8.2), facecolor=BG)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.15, 1.0, 0.95], hspace=0.28, left=0.09, right=0.97, top=0.90, bottom=0.08)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    for ax in (ax0, ax1, ax2):
        ax.set_facecolor(BG)
        ax.grid(True, color=C_LINE, lw=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- panel 1: shape fractions (stacked) ---
    if not state.empty:
        t = state["t_mid"]
        bottoms = np.zeros(len(state))
        for sh in SHAPE_ORDER:
            y = state[f"frac_{sh}"].to_numpy(float)
            y = np.nan_to_num(y, nan=0.0)
            ax0.fill_between(t, bottoms, bottoms + y, color=SHAPE_COLORS[sh], alpha=0.85, lw=0, label=sh, step="mid")
            bottoms = bottoms + y
        ax0.set_ylim(0, 1.02)
        ax0.set_ylabel("Shape fraction", color=C_INK)
        ax0.legend(loc="upper left", ncol=5, fontsize=7.2, frameon=False, bbox_to_anchor=(0.0, 1.18))
        # labeled n as dots
        ax0b = ax0.twinx()
        ax0b.plot(t, state["n_labeled"], "o-", color=C_MUTED, ms=3.5, lw=0.8, alpha=0.8)
        ax0b.set_ylabel("n labeled", color=C_MUTED, fontsize=8)
        ax0b.tick_params(axis="y", labelsize=7, colors=C_MUTED)
        ax0b.spines["top"].set_visible(False)

    # --- panel 2: β_res ---
    if not state.empty:
        ax1.plot(state["t_mid"], state["beta_median"], color="#0F766E", lw=1.6, label=r"median $\beta_{\mathrm{res}}$")
        ax1.fill_between(state["t_mid"], state["beta_mean"], state["beta_median"], color="#0F766E", alpha=0.12, lw=0)
        ax1.axhline(0.0, color=C_MUTED, lw=0.7, ls="--")
        ax1.set_ylabel(r"Coda path residual $\beta_{\mathrm{res}}$", color=C_INK)
        ax1.legend(loc="upper right", fontsize=8, frameon=False)

    # strong-event markers
    if not strong.empty:
        for _, ev in strong.iterrows():
            for ax in (ax0, ax1, ax2):
                ax.axvline(ev["t"], color="#B91C1C", lw=0.9, alpha=0.55, zorder=1)
        # annotate a few largest
        top = strong.nlargest(min(5, len(strong)), "mag")
        for _, ev in top.iterrows():
            ax1.annotate(
                f"M{ev['mag']:.1f}",
                xy=(ev["t"], ax1.get_ylim()[1] if ax1.get_ylim()[1] != ax1.get_ylim()[0] else 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=6.5,
                color="#B91C1C",
                ha="center",
                va="bottom",
            )

    # --- panel 3: activity ---
    if not act.empty:
        ax2.bar(act["t_mid"], act["n_mod"], width=20, color="#64748B", alpha=0.55, label=f"n events M≥{meta.get('mag_mod', 3):.0f}")
        ax2.plot(act["t_mid"], act["n_strong"], "D-", color="#B91C1C", ms=4, lw=1.0, label=f"n events M≥{meta.get('mag_strong', 5):.0f}")
        ax2.set_ylabel("STEAD unique events / bin", color=C_INK)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

    ax2.set_xlabel("Time", color=C_INK)
    rho = corr.get("beta_vs_log_nmod", {})
    fig.suptitle(
        f"{meta['title']}: morphology / β_res evolution vs STEAD seismicity\n"
        f"Spearman(β_res, log1p M≥mod n) ρ={rho.get('rho', float('nan')):.2f} "
        f"(n_bins={rho.get('n', 0)}, p≈{rho.get('p', float('nan')):.3g})  |  "
        f"strong-event windows tested={len(window_tests)}",
        fontsize=11,
        color=C_INK,
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=BG)
    fig.savefig(out_png.with_suffix(".pdf"), dpi=dpi, facecolor=BG)
    plt.close(fig)


def analyze_region(
    key: str,
    labeled_all: pd.DataFrame,
    cat_all: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    meta = dict(REGIONS[key])
    meta["mag_mod"] = args.mag_mod
    meta["mag_strong"] = args.mag_strong
    lat_rng, lon_rng = meta["lat"], meta["lon"]

    lab = labeled_all[
        labeled_all["src_lat"].between(*lat_rng) & labeled_all["src_lon"].between(*lon_rng)
    ].copy()
    print(f"[{key}] labeled in box: {len(lab)}", flush=True)

    cat = filter_box(cat_all, lat_rng, lon_rng)
    print(f"[{key}] STEAD unique events in box: {len(cat)}", flush=True)

    state = binned_state(lab, args.freq, args.min_bin_n)
    act = binned_activity(cat, args.freq, args.mag_mod, args.mag_strong)

    # align bins for correlation
    corr = {}
    if not state.empty and not act.empty:
        m = state.merge(act[["period", "n_mod", "n_strong", "n_events"]], on="period", how="inner")
        corr["beta_vs_log_nmod"] = spearman(
            m["beta_median"].to_numpy(float),
            np.log1p(m["n_mod"].to_numpy(float)),
        )
        corr["frac_impulsive_vs_log_nmod"] = spearman(
            m["frac_impulsive"].to_numpy(float),
            np.log1p(m["n_mod"].to_numpy(float)),
        )
        corr["frac_slow_vs_log_nmod"] = spearman(
            m["frac_slow_coda"].to_numpy(float),
            np.log1p(m["n_mod"].to_numpy(float)),
        )
        corr["beta_vs_log_nstrong"] = spearman(
            m["beta_median"].to_numpy(float),
            np.log1p(m["n_strong"].to_numpy(float)),
        )

    strong = cat[cat["mag"] >= args.mag_strong].copy() if not cat.empty else cat
    # keep strong events inside labeled time span (±1y pad) to make windows meaningful
    if not lab.empty and not strong.empty:
        t0, t1 = lab["t"].min() - pd.Timedelta(days=args.pre_days), lab["t"].max() + pd.Timedelta(days=args.post_days)
        strong = strong[(strong["t"] >= t0) & (strong["t"] <= t1)]
    windows = event_window_tests(lab, strong, args.pre_days, args.post_days)

    out_dir = Path(args.output_dir) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    if not state.empty:
        state.to_csv(out_dir / "state_bins.csv", index=False)
    if not act.empty:
        act.to_csv(out_dir / "activity_bins.csv", index=False)
    if not strong.empty:
        strong.to_csv(out_dir / "strong_events.csv", index=False)
    if windows:
        pd.DataFrame(windows).to_csv(out_dir / "strong_event_windows.csv", index=False)

    plot_region(
        key,
        meta,
        state,
        act,
        strong,
        windows,
        corr,
        out_dir / f"{key}_temporal_evolution.png",
        args.dpi,
    )

    summary = {
        "region": key,
        "title": meta["title"],
        "box": {"lat": list(lat_rng), "lon": list(lon_rng)},
        "note": meta["note"],
        "n_labeled": int(len(lab)),
        "n_stead_events": int(len(cat)),
        "n_strong_in_span": int(len(strong)),
        "freq": args.freq,
        "correlations": corr,
        "window_tests_n": len(windows),
        "window_delta_beta_median": float(np.nanmedian([w["delta_beta"] for w in windows])) if windows else None,
        "window_frac_impulsive_delta_median": (
            float(np.nanmedian([w["frac_impulsive_post"] - w["frac_impulsive_pre"] for w in windows])) if windows else None
        ),
        "year_range_labeled": (
            [str(lab["t"].min().date()), str(lab["t"].max().date())] if not lab.empty else None
        ),
        "caveats": [
            "Labeled shape/β series come from the interpretable ceiling subset (sparse), not full STEAD.",
            "STEAD activity is a selected waveform archive, not a complete regional catalog — rates are relative.",
            "Network densification can confound long-term fraction trends; prefer short windows around strong events.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{key}] wrote {out_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    labeled = load_labeled(Path(args.traces))
    if args.fixed_only:
        if "fixed_station" not in labeled.columns:
            raise SystemExit("--fixed-only requires a fixed_station column in traces CSV")
        before = len(labeled)
        labeled = labeled[labeled["fixed_station"]].copy()
        print(f"[all] fixed-only filter: {before} → {len(labeled)}", flush=True)
    print(f"[all] labeled traces: {len(labeled)}  years {labeled['t'].dt.year.min()}–{labeled['t'].dt.year.max()}", flush=True)

    keys = [k.strip() for k in args.regions.split(",") if k.strip()]
    unknown = [k for k in keys if k not in REGIONS]
    if unknown:
        raise SystemExit(f"unknown regions {unknown}; choose from {sorted(REGIONS)}")

    cat_all = load_stead_events_union(Path(args.stead_dir), keys)
    print(f"[all] STEAD unique events in union boxes: {len(cat_all)}", flush=True)

    summaries = []
    for key in keys:
        summaries.append(analyze_region(key, labeled, cat_all, args))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary_all.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # concise markdown report
    lines = [
        "# Shape / β_res temporal evolution vs STEAD seismicity",
        "",
        f"- Labeled traces: `{args.traces}` (n={len(labeled)})",
        f"- Binning: `{args.freq}`; moderate M≥{args.mag_mod}, strong M≥{args.mag_strong}",
        f"- Pre/post windows: {args.pre_days:.0f}d / {args.post_days:.0f}d (excluding ±7d)",
        "",
    ]
    for s in summaries:
        c = s["correlations"].get("beta_vs_log_nmod", {})
        lines += [
            f"## {s['title']} (`{s['region']}`)",
            f"- Labeled in box: **{s['n_labeled']}**; STEAD unique events: **{s['n_stead_events']}**; strong in span: **{s['n_strong_in_span']}**",
            f"- Spearman(β_res, log1p M≥mod): ρ={c.get('rho', float('nan')):.3f}, p≈{c.get('p', float('nan')):.3g}, bins={c.get('n', 0)}",
            f"- Strong-event windows tested: **{s['window_tests_n']}**; median Δβ_res={s['window_delta_beta_median']}",
            f"- Figure: `{s['region']}/{s['region']}_temporal_evolution.png`",
            "",
        ]
    lines += [
        "## Caveats",
        "- Shape/β time series inherit the sparse ceiling-labeled subset.",
        "- STEAD rates are archive-relative, not complete catalog rates.",
        "- Long-term trends may track network growth; event-window contrasts are more credible.",
        "",
    ]
    (out_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out_root / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
