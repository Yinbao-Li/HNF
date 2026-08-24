#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Figs. 1–6 for the unified-propagation article SI."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
BG = "#FFFFFF"
C_SALTON = "#D55E00"
C_ETR = "#0072B2"
C_OK = "#009E73"
C_MP = "#009E73"
C_SC = "#E69F00"
C_CTRL = "#9CA3AF"
C_WARN = "#F59E0B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/figures/unified")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--only", default="all", help="Comma list: 1,2,3,4,5,6 or all")
    return p.parse_args()


def resolve(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else _REPO / p


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.linewidth": 0.7,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, colors=C_INK)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK)
        ax.spines[side].set_linewidth(0.7)


def hdr(ax, letter: str, title: str):
    """Legacy empty-strip header (prefer tag() on plot axes)."""
    ax.set_axis_off()
    ax.text(0.0, 0.35, letter, fontsize=11, fontweight="bold", color=C_INK, va="center", transform=ax.transAxes)
    ax.text(0.055, 0.35, title, fontsize=7.2, color=C_INK, va="center", transform=ax.transAxes)


def tag(ax, letter: str, title: str, *, y: float = 1.01):
    """Put panel letter+title tightly above a plot axes (no separate header row)."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=C_INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.text(
        0.055,
        y,
        title,
        transform=ax.transAxes,
        fontsize=7.2,
        color=C_INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def block_gs(fig, outer, *, nrows=2, ncols=1, title_ratio=0.07, hspace=0.04, wspace=0.28):
    """Nested gridspec: thin title row + plot row(s), local tight spacing."""
    return GridSpecFromSubplotSpec(
        nrows,
        ncols,
        subplot_spec=outer,
        height_ratios=[title_ratio, 1.0] if nrows == 2 else None,
        hspace=hspace,
        wspace=wspace,
    )


def save_fig(fig, out_dir: Path, stem: str, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"[ed] wrote {png.relative_to(_REPO)}")
    print(f"[ed] wrote {pdf.relative_to(_REPO)}")


def cell_mask(df: pd.DataFrame, which: str) -> np.ndarray:
    if which == "salton":
        return (df["gx"] == -116.375) & (df["gy"] == 33.375)
    if which == "etr":
        return (df["gx"] == -116.375) & (df["gy"] == 33.875)
    raise ValueError(which)


def _load_shape_geo_mod():
    path = _REPO / "tools/plot_shape_geo_coda_figure.py"
    spec = importlib.util.spec_from_file_location("plot_shape_geo_coda_figure", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _jk_compact(tests: list[dict]) -> list[dict]:
    keep_names = {
        "full",
        "drop_top1_srcbins",
        "drop_top5_srcbins",
        "drop_top10_srcbins",
        "mag_floor_2.0",
        "mag_floor_2.5",
        "cap_srcbin_5",
        "cap_srcbin_10",
        "cap10_per_srcbin",
        "cap20_per_srcbin",
        "cap40_per_srcbin",
    }
    keep = []
    years = []
    for t in tests:
        name = t["test"]
        if name.startswith("drop_year_"):
            years.append(t)
        elif name in keep_names:
            keep.append(t)
    if years:
        keep.append(
            {
                "test": "year_LOO_median",
                "site_delta_med": float(np.median([t["site_delta_med"] for t in years])),
                "mw_p": float(np.median([t["mw_p"] for t in years])),
                "frac_neg": float(np.median([t.get("frac_neg", np.nan) for t in years])),
            }
        )
    return keep


def _jk_label(name: str) -> str:
    return (
        name.replace("drop_", "")
        .replace("_srcbins", "")
        .replace("mag_floor_", "M≥")
        .replace("cap_srcbin_", "cap")
        .replace("cap", "cap")
        .replace("_per_srcbin", "")
        .replace("year_LOO_median", "year LOO")
    )


# ---------------------------------------------------------------------------
# ED1 — map + class characters only
# ---------------------------------------------------------------------------

def plot_ed1(out_dir: Path, dpi: int):
    geo = _load_shape_geo_mod()
    geo._style()
    traces = _REPO / "outputs/interpretable_physics_best/ceiling/traces_labeled.csv"
    coast_path = _REPO / "docs/figures/geo/ne_110m_land.geojson"
    means_path = _REPO / "docs/figures/seismic_shape_geo_coda_means.npz"

    df = pd.read_csv(traces)
    # local-earthquake test geography: drop noise if present
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower().eq("test")].copy()
    if "shape" in df.columns:
        df = df[df["shape"].isin(geo.SHAPE_ORDER)].copy()
    coast = geo._load_coast_polygons(coast_path)
    means = geo._load_means(means_path)
    if means is None:
        raise FileNotFoundError(f"missing means cache: {means_path}")

    fig = plt.figure(figsize=(7.4, 8.4))
    # Outer: two content blocks with clear gap; titles sit tight on each axes.
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.55, 1.05], hspace=0.34, left=0.06, right=0.98, top=0.95, bottom=0.04)

    ax_map = fig.add_subplot(gs[0, 0])
    geo.plot_panel_b_map(ax_map, df, coast)
    tag(ax_map, "a", "Geographic enrichment of five operational facies (STEAD test)")

    ax_char = fig.add_subplot(gs[1, 0])
    geo.plot_panel_d_characters(ax_char, df, means)
    tag(ax_char, "b", r"Morphology-class characters ($\rho$ shape, $\beta_{\mathrm{res}}$, enrichment, Q)")

    save_fig(fig, out_dir, "edfig1_faciesgeo", dpi)


# ---------------------------------------------------------------------------
# ED2 — independence (no annotation occlusion)
# ---------------------------------------------------------------------------

def plot_ed2(out_dir: Path, dpi: int):
    qc = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_classical_qc.csv")
    berg = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_berg2021.csv")
    lin = pd.read_csv(_REPO / "outputs/structure_residual_socal/beta_vs_lin2023_q.csv")
    qc_sum = load_json(_REPO / "outputs/structure_residual_socal/beta_vs_classical_qc_summary.json")
    berg_sum = load_json(_REPO / "outputs/structure_residual_socal/beta_vs_berg2021_summary.json")
    lin_sum = load_json(_REPO / "outputs/structure_residual_socal/beta_vs_lin2023_q_summary.json")

    style()
    fig = plt.figure(figsize=(7.4, 7.8))
    # Three content rows with large inter-row gap; titles tagged on axes.
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.05, 1.05, 0.48],
        hspace=0.48,
        wspace=0.32,
        left=0.09,
        right=0.98,
        top=0.95,
        bottom=0.08,
    )

    # a
    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    tag(ax, "a", rf"Classical single-trace $q_c$   $r={qc_sum['pearson_beta_vs_qc']:.2f}$")
    ax.scatter(qc["qc_slope"], qc["beta_resid"], s=3, c=C_CTRL, alpha=0.22, rasterized=True, linewidths=0)
    for name, color, which in (("Salton", C_SALTON, "salton"), ("ETR", C_ETR, "etr")):
        m = cell_mask(qc, which)
        ax.scatter(qc.loc[m, "qc_slope"], qc.loc[m, "beta_resid"], s=9, c=color, alpha=0.8, label=name, linewidths=0)
    x = qc["qc_slope"].to_numpy()
    y = qc["beta_resid"].to_numpy()
    coef = np.polyfit(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)], 1)
    xx = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 80)
    ax.plot(xx, np.polyval(coef, xx), color=C_INK, lw=1.1)
    ax.set_xlabel(r"classical coda decay slope (more negative = faster)")
    ax.set_ylabel(r"$\beta_{\mathrm{res}}$")
    ax.set_ylim(-0.22, 0.22)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", borderaxespad=0.2)

    # b
    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    tag(ax, "b", r"Facies still split inside matched $q_c$ terciles")
    bins = qc_sum["within_qc_shape_split"]
    xpos = np.arange(len(bins))
    w = 0.34
    ax.bar(xpos - w / 2, [b["beta_mp"] for b in bins], width=w, color=C_MP, label="multipath")
    ax.bar(xpos + w / 2, [b["beta_sc"] for b in bins], width=w, color=C_SC, label="slow coda")
    ax.axhline(0, color=C_LINE, lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([b["qc_bin"] for b in bins])
    ax.set_ylabel(r"median $\beta_{\mathrm{res}}$")
    ax.legend(frameon=False, fontsize=6.0, loc="upper right")

    # c
    band = berg_sum["vs_overlap_split"]["vs_band"]
    dlt = berg_sum["vs_overlap_split"]["delta_#5_minus_#2"]
    ax = fig.add_subplot(gs[1, 0])
    clean(ax)
    tag(ax, "c", rf"Berg 2021 $V_S$ · SPLIT  $r={berg_sum['pearson_beta_vs_0to8']:.2f}$")
    ax.scatter(berg["vs_0to8"], berg["beta_resid"], s=3, c=C_CTRL, alpha=0.18, rasterized=True, linewidths=0)
    for name, color, which in (("Salton", C_SALTON, "salton"), ("ETR", C_ETR, "etr")):
        m = cell_mask(berg, which)
        ax.scatter(berg.loc[m, "vs_0to8"], berg.loc[m, "beta_resid"], s=9, c=color, alpha=0.85, label=name, linewidths=0)
    ax.axvspan(band[0], band[1], color="#FDE68A", alpha=0.40, zorder=0)
    ax.text(
        0.03,
        0.97,
        rf"shared $V_S$ band" + "\n" + rf"$\Delta\beta={dlt:+.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=C_INK,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_LINE, lw=0.6, alpha=0.92),
    )
    ax.set_xlabel(r"path-midpoint mean $V_S$ 0–8 km (km s$^{-1}$)")
    ax.set_ylabel(r"$\beta_{\mathrm{res}}$")
    ax.set_ylim(-0.22, 0.22)
    ax.legend(frameon=False, fontsize=6.0, loc="lower left")

    # d
    ax = fig.add_subplot(gs[1, 1])
    clean(ax)
    tag(ax, "d", r"Lin \& Jordan 2023 $Q_S$ · orthogonal / anti-aligned")
    ax.scatter(lin["qs_0to8"], lin["beta_resid"], s=3, c=C_CTRL, alpha=0.15, rasterized=True, linewidths=0)
    for name, color, which in (("Salton", C_SALTON, "salton"), ("ETR", C_ETR, "etr")):
        m = cell_mask(lin, which)
        ax.scatter(lin.loc[m, "qs_0to8"], lin.loc[m, "beta_resid"], s=9, c=color, alpha=0.85, label=name, linewidths=0)
    ax.annotate(
        "high $Q_S$\nyet faster coda",
        xy=(0.92, 0.28),
        xycoords="axes fraction",
        xytext=(0.55, 0.08),
        textcoords="axes fraction",
        fontsize=5.6,
        color=C_SALTON,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=C_SALTON, lw=0.8, connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
    )
    ax.annotate(
        "lower $Q_S$\nyet rings",
        xy=(0.42, 0.62),
        xycoords="axes fraction",
        xytext=(0.18, 0.90),
        textcoords="axes fraction",
        fontsize=5.6,
        color=C_ETR,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color=C_ETR, lw=0.8, connectionstyle="arc3,rad=-0.12"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
    )
    ax.set_xlabel(r"path-midpoint mean $Q_S$ 0–8 km")
    ax.set_ylabel(r"$\beta_{\mathrm{res}}$")
    ax.set_ylim(-0.22, 0.22)
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")

    # leftover Δβ footer
    foot = fig.add_subplot(gs[2, :])
    clean(foot)
    c2 = next(c for c in qc_sum["cells"] if c["rank"] == 2)
    c5 = next(c for c in qc_sum["cells"] if c["rank"] == 5)
    b2 = next(c for c in berg_sum["cells"] if c["rank"] == 2)
    b5 = next(c for c in berg_sum["cells"] if c["rank"] == 5)
    q2 = next(c for c in lin_sum["cells"] if c["rank"] == 2)
    q5 = next(c for c in lin_sum["cells"] if c["rank"] == 5)
    labels = [r"after $q_c$", r"after $V_S$", r"after $Q_S$"]
    sal_v = [c2["delta_beta_after_qc"], b2["delta_beta_after_vs"], q2["delta_beta_after_qs"]]
    etr_v = [c5["delta_beta_after_qc"], b5["delta_beta_after_vs"], q5["delta_beta_after_qs"]]
    x = np.arange(3)
    w = 0.34
    foot.axhline(0, color=C_LINE, lw=0.8)
    foot.bar(x - w / 2, sal_v, width=w, color=C_SALTON, label="Salton")
    foot.bar(x + w / 2, etr_v, width=w, color=C_ETR, label="ETR")
    foot.set_xticks(x)
    foot.set_xticklabels(labels)
    foot.set_ylabel(r"leftover $\Delta\beta$")
    foot.set_title(
        "Leftover cell contrast after removing each observable",
        loc="left",
        fontsize=7.2,
        color=C_INK,
        pad=3,
    )
    y_hi = max(max(sal_v), max(etr_v), 0.0)
    y_lo = min(min(sal_v), min(etr_v), 0.0)
    pad = 0.08 * max(y_hi - y_lo, 0.02)
    foot.set_ylim(y_lo - pad, y_hi + pad)
    foot.legend(
        frameon=False,
        fontsize=5.8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
    )

    save_fig(fig, out_dir, "edfig2_independence", dpi)


# ---------------------------------------------------------------------------
# ED3 — Cascadia map large + visual jackknives
# ---------------------------------------------------------------------------

def plot_ed3(out_dir: Path, dpi: int):
    style()
    cascadia_png = _REPO / "outputs/structure_residual_cascadia_volc/cascadia_volc_structure_residuals.png"
    jk = load_json(_REPO / "outputs/structure_residual_socal/socal_cell_jackknife.json")
    st_csv = pd.read_csv(_REPO / "outputs/structure_residual_cascadia_volc/sthelens_source_jackknife.csv")
    st = load_json(_REPO / "outputs/structure_residual_cascadia_volc/sthelens_source_jackknife.json")

    # Fill cell with aspect='auto'; crop source title banner (we add panel tag ourselves).
    img = _crop_whitespace(plt.imread(cascadia_png))
    # drop embedded figure title / subtitle strip at top of source PNG
    img = img[int(0.085 * img.shape[0]) :, :]
    fig = plt.figure(figsize=(7.6, 10.2))
    gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[3.4, 1.0, 1.0],
        hspace=0.05,
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.05,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    tag(ax, "a", r"Cascadia volcanic arc — structure residual of $\beta_{\mathrm{res}}$ and facies", y=1.005)

    ax = fig.add_subplot(gs[1, 0])
    clean(ax)
    tag(ax, "b", "SoCal same-station jackknife (Salton absorbing / ETR ringing)")
    sal = _jk_compact(jk["cells"]["salton"]["tests"])
    etr = _jk_compact(jk["cells"]["etr"]["tests"])
    etr_map = {t["test"]: t["site_delta_med"] for t in etr}
    labels = [_jk_label(t["test"]) for t in sal]
    x = np.arange(len(labels))
    ax.axhline(0, color=C_LINE, lw=0.9)
    ax.fill_between(x, -0.01, 0.01, color="#F3F4F6", alpha=0.7, zorder=0)
    ax.plot(x, [t["site_delta_med"] for t in sal], "o-", color=C_SALTON, ms=5, lw=1.2, label="Salton")
    ax.plot(x, [etr_map.get(t["test"], np.nan) for t in sal], "s-", color=C_ETR, ms=5, lw=1.2, label="ETR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=5.6)
    ax.set_ylabel(r"site-median $\Delta\beta$")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", ncol=2)
    ax.text(
        0.01,
        0.97,
        "verdict: STABLE / STABLE",
        transform=ax.transAxes,
        fontsize=6.2,
        color=C_OK,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#A7F3D0", lw=0.7),
    )

    ax = fig.add_subplot(gs[2, 0])
    clean(ax)
    tag(ax, "c", "Cascadia jackknife — Mt St Helens–Cowlitz (Salton-signed replica)")
    compact = _jk_compact(st_csv.to_dict("records"))
    labs = [_jk_label(t["test"]) for t in compact]
    ys = [t["site_delta_med"] for t in compact]
    x = np.arange(len(labs))
    ax.axhline(0, color=C_LINE, lw=0.9)
    ax.axhline(st["full"]["site_delta_med"], color=C_SALTON, lw=0.9, ls="--", alpha=0.7)
    ax.bar(x, ys, color=C_OK, width=0.72, edgecolor="white", linewidth=0.4, zorder=2)
    ax.plot(x, ys, "o", color=C_INK, ms=3.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=28, ha="right", fontsize=5.6)
    ax.set_ylabel(r"site-median $\Delta\beta$")
    ax.set_xlim(-0.6, len(labs) - 0.4)
    ymin, ymax = ax.get_ylim()
    band_y = ymax + 0.08 * (ymax - ymin)
    for i, yv in enumerate(ys):
        ax.plot(
            i,
            band_y,
            marker="s",
            ms=7,
            color=C_OK if yv < 0 else C_SALTON,
            markeredgecolor="white",
            markeredgewidth=0.4,
            clip_on=False,
            zorder=5,
        )
    ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
    ax.text(-0.55, band_y, "sign", fontsize=5.4, color=C_MUTED, va="center", ha="right", clip_on=False)
    ax.text(
        0.01,
        0.08,
        f"sign-stable {st['n_sign_stable']}/{st['n_tests']} · year LOO all neg · {st['verdict']}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=C_OK,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#A7F3D0", lw=0.6, alpha=0.92),
    )

    save_fig(fig, out_dir, "edfig3_cascadia", dpi)


# ---------------------------------------------------------------------------
# ED4 — classical control (visual R² panel)
# ---------------------------------------------------------------------------

def plot_ed4(out_dir: Path, dpi: int):
    style()
    tr = pd.read_csv(_REPO / "outputs/structure_residual_socal/pnf_vs_classical_coda/traces_pnf_vs_classical.csv")
    summ = load_json(_REPO / "outputs/structure_residual_socal/pnf_vs_classical_coda/SUMMARY.json")

    fig = plt.figure(figsize=(7.4, 6.8))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.15, 1.0],
        hspace=0.42,
        wspace=0.30,
        left=0.10,
        right=0.98,
        top=0.94,
        bottom=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    tag(ax, "a", rf"Probe vs catalog $\beta_{{\mathrm{{res}}}}$  ($r={summ['corr_beta_resid_pnf_vs_classical']:.3f}$)")
    ax.scatter(tr["beta_resid_classical"], tr["beta_resid"], s=4, c=C_CTRL, alpha=0.25, rasterized=True, linewidths=0)
    lim = np.nanpercentile(np.r_[tr["beta_resid"], tr["beta_resid_classical"]], [1, 99])
    ax.plot(lim, lim, color=C_INK, lw=0.9, ls="--")
    ax.set_xlabel(r"catalog-timed classical $\beta_{\mathrm{res}}$")
    ax.set_ylabel(r"probe-timed $\beta_{\mathrm{res}}$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_box_aspect(1)

    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    tag(ax, "b", rf"Coda slope agreement  ($r={summ['corr_coda_slope_pnf_vs_catalog']:.3f}$)")
    ax.scatter(tr["classical_coda_slope"], tr["coda_slope"], s=4, c=C_CTRL, alpha=0.25, rasterized=True, linewidths=0)
    lim = np.nanpercentile(np.r_[tr["coda_slope"], tr["classical_coda_slope"]], [1, 99])
    ax.plot(lim, lim, color=C_INK, lw=0.9, ls="--")
    ax.set_xlabel("catalog-timed classical coda slope")
    ax.set_ylabel("probe-timed coda slope")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_box_aspect(1)

    ax = fig.add_subplot(gs[1, 0])
    clean(ax)
    tag(ax, "c", r"Same-station cell $\Delta\beta$ (classical vs probe)")
    cells = summ["cell_same_station"]
    names = []
    for c in cells:
        if c["cell"] not in names:
            names.append(c["cell"])
    x = np.arange(len(names))
    w = 0.34
    clas = [next(c["median_delta_beta"] for c in cells if c["cell"] == n and c["pipeline"] == "classical") for n in names]
    pnf = [next(c["median_delta_beta"] for c in cells if c["cell"] == n and c["pipeline"] == "pnf") for n in names]
    ax.axhline(0, color=C_LINE, lw=0.9)
    ax.bar(x - w / 2, clas, width=w, color="#94A3B8", label="catalog-timed")
    ax.bar(x + w / 2, pnf, width=w, color=C_OK, label="probe-timed")
    ax.set_xticks(x)
    ax.set_xticklabels(["Salton", "ETR"])
    ax.set_ylabel(r"site-median $\Delta\beta$")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    clean(ax)
    tag(ax, "d", rf"OOS $R^2$ to predict probe $\beta$  ($n={summ['n_traces']}$)")
    r2 = summ["oos_r2_predict_pnf_beta"]
    vals = [r2["classical_only"], r2["classical_plus_pnf_latents"]]
    labels = ["classical\nonly", "classical\n+ probe latents"]
    bars = ax.bar([0, 1], vals, color=[C_ETR, "#94A3B8"], width=0.55, edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"OOS $R^2$")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=7.0, color=C_INK)
    dlt = r2["delta_r2"]
    ax.annotate(
        "",
        xy=(1.0, vals[1]),
        xytext=(0.0, vals[0]),
        arrowprops=dict(arrowstyle="<->", color=C_MUTED, lw=1.0),
    )
    ax.text(
        0.5,
        0.92,
        rf"$\Delta R^2={dlt:+.4f}$  (no incremental gain)",
        ha="center",
        fontsize=6.2,
        color=C_MUTED,
        transform=ax.transAxes,
    )

    save_fig(fig, out_dir, "edfig4_classical", dpi)


# ---------------------------------------------------------------------------
# ED5 — continuum (visual corner recovery)
# ---------------------------------------------------------------------------

def plot_ed5(out_dir: Path, dpi: int):
    style()
    path = load_json(_REPO / "outputs/propagation_dynamics/unified_theta_lambda_v1/wave_to_diffusion.json")
    agg = load_json(_REPO / "outputs/propagation_dynamics/unified_theta_lambda_v1/AGGREGATE.json")

    rows = path["rows"]
    lam = np.array([r["lambda"] for r in rows], float)
    lag = np.array([r["mean_lag"] for r in rows], float)
    phases = [r["phase"] for r in rows]
    damp = 4.0 * lam * (1.0 - lam)

    phase_color = {
        "wave_like": "#0072B2",
        "damped_wave": "#CC79A7",
        "diffusive": "#009E73",
        "instantaneous_like": "#E69F00",
        "transitional": "#94A3B8",
    }

    # Drop former panel d; a|b on top, full-width phase sequence below.
    fig = plt.figure(figsize=(7.4, 6.0))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.35, 0.78],
        hspace=0.50,
        wspace=0.32,
        left=0.10,
        right=0.98,
        top=0.92,
        bottom=0.10,
    )

    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    tag(ax, "a", r"Primary path wave$\rightarrow$diffusion: mean lag vs $\lambda$")
    for i in range(len(lam) - 1):
        ax.plot(lam[i : i + 2], lag[i : i + 2], color=phase_color.get(phases[i], C_MUTED), lw=2.0)
    ax.scatter(
        lam,
        lag,
        c=[phase_color.get(p, C_MUTED) for p in phases],
        s=22,
        zorder=3,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("mean lag")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, max(lag) * 1.12)
    ax.text(
        0.98,
        0.95,
        "monotone lag collapse\ndynamical_space_pass",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.6,
        color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_LINE, lw=0.5, alpha=0.9),
    )

    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    tag(ax, "b", r"Telegrapher damping proxy $\zeta=4\lambda(1-\lambda)$")
    ax.plot(lam, damp, color="#CC79A7", lw=1.8)
    ax.fill_between(lam, 0, damp, color="#CC79A7", alpha=0.15)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"damping proxy $\zeta$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.15)
    ax.annotate(
        "mid-path peak\n= damped-wave",
        xy=(0.5, 1.0),
        xytext=(0.72, 0.55),
        fontsize=5.8,
        color=C_MUTED,
        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=0.7),
        ha="center",
    )

    ax = fig.add_subplot(gs[1, :])
    clean(ax)
    used = []
    for ph in phases:
        if ph not in used:
            used.append(ph)
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=phase_color[p], markersize=7, label=p.replace("_", " "))
        for p in used
        if p in phase_color
    ]
    tag(ax, "c", r"Phase sequence along $\Theta(\lambda)$")
    for i, ph in enumerate(phases):
        ax.barh(0, 1.0, left=i, height=0.55, color=phase_color.get(ph, C_MUTED), edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, len(phases))
    ax.set_ylim(-0.75, 0.85)
    ax.set_yticks([])
    ax.set_xlabel(r"step along $\Theta(\lambda)$")
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=5.6,
        loc="upper left",
        ncol=min(4, len(handles)),
        bbox_to_anchor=(0.22, 1.0),
        handletextpad=0.25,
        columnspacing=0.7,
    )
    ax.text(
        0.01,
        0.05,
        f"primary={agg['primary_path']} · visits_damped_wave={agg['visits_damped_wave']}",
        transform=ax.transAxes,
        fontsize=5.4,
        color=C_MUTED,
        va="bottom",
    )

    save_fig(fig, out_dir, "edfig5_unified", dpi)


# ---------------------------------------------------------------------------
# ED6 — SST RDG (visual scale caveat)
# ---------------------------------------------------------------------------

def softplus(x):
    x = np.asarray(x, float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _crop_whitespace(img: np.ndarray, thr: float = 0.985, pad: int = 4) -> np.ndarray:
    """Trim near-white margins from an RGB(A) image array."""
    if img.ndim == 3:
        gray = img[..., :3].mean(axis=2)
    else:
        gray = img
    if gray.max() > 1.5:
        gray = gray / 255.0
    mask = gray < thr
    if not np.any(mask):
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ridx = np.where(rows)[0]
    cidx = np.where(cols)[0]
    r0 = max(int(ridx[0]) - pad, 0)
    r1 = min(int(ridx[-1]) + pad + 1, img.shape[0])
    c0 = max(int(cidx[0]) - pad, 0)
    c1 = min(int(cidx[-1]) + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def plot_ed6(out_dir: Path, dpi: int):
    style()
    phys = load_json(_REPO / "outputs/propagation_dynamics/sst_rdg_physics_v1/physics_map.json")
    b0, b1, b2 = phys["b0"], phys["b1"], phys["b2"]
    r = np.linspace(0.0, 2.2, 280)
    g = softplus(b0 + b1 * r + b2 * r * r)
    km_slope = phys["km_per_rhat_slope"]

    fig = plt.figure(figsize=(7.4, 6.4))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.2, 1.0],
        hspace=0.45,
        wspace=0.32,
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    tag(ax, "a", r"Frozen RDG gain $g(\hat r)$ on the SST sensor graph")
    ax.plot(r, g, color=C_OK, lw=2.0)
    ax.axhline(1.0, color=C_LINE, lw=0.9, ls="--")
    rmin = phys["rhat_at_g_min"]
    ax.axvline(rmin, color=C_MUTED, lw=0.8, ls=":")
    ax.scatter([rmin], [phys["g_min"]], s=40, c=C_SALTON, zorder=3)
    ax.annotate(
        rf"$g_{{\min}}$ @ $\hat r={rmin:.2f}$" + "\n" + rf"~{phys['km_at_g_min']:.0f} km on graph",
        xy=(rmin, phys["g_min"]),
        xytext=(0.62, 0.82),
        textcoords="axes fraction",
        fontsize=5.8,
        color=C_MUTED,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_LINE, lw=0.5, alpha=0.92),
    )
    ax.set_xlabel(r"normalised range $\hat r$")
    ax.set_ylabel(r"$g(\hat r)$")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0.55, max(g) * 1.05)

    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    tag(ax, "b", r"Consistency: $g$ vs $|\Delta\mathrm{SST}|$ by band")
    bands = ["near", "mid", "far"]
    g_vals = [phys["check"][f"g_{b}"] for b in bands]
    d_vals = [phys["check"][f"abs_dSST_{b}"] for b in bands]
    x = np.arange(3)
    ax.bar(x - 0.18, g_vals, width=0.34, color=C_OK, label=r"$g$")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, d_vals, width=0.34, color="#94A3B8", label=r"$|\Delta\mathrm{SST}|$")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel(r"mean $g$", color=C_OK)
    ax2.set_ylabel(r"mean $|\Delta\mathrm{SST}|$", color=C_MUTED)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax.tick_params(axis="y", colors=C_OK)
    ax2.tick_params(axis="y", colors=C_MUTED)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.8, loc="upper right")
    ax.text(
        0.02,
        0.05,
        f"phys_pass · corr={phys['check']['corr_g_vs_abs']:+.2f}",
        transform=ax.transAxes,
        fontsize=5.6,
        color=C_MUTED,
    )

    ax = fig.add_subplot(gs[1, :])
    clean(ax)
    tag(ax, "c", "Sensor-graph scale caveat — mechanism bands vs physical length scales")
    near_km = (0.0 * km_slope, 0.5 * km_slope)
    mid_km = (0.9 * km_slope, 1.7 * km_slope)
    far_km = (1.7 * km_slope, 2.2 * km_slope)
    rossby = 50.0

    ax.set_xscale("log")
    ax.set_xlim(20, 40000)
    ax.set_ylim(0, 3.2)
    ax.set_yticks([])
    ax.set_xlabel("length scale on this embedding (km, log)")

    def band_bar(y, lo, hi, color, label, sub):
        ax.add_patch(Rectangle((lo, y - 0.28), hi - lo, 0.56, facecolor=color, edgecolor="white", lw=0.6, alpha=0.85))
        ax.text(np.sqrt(lo * hi), y, label, ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")
        ax.text(hi * 1.05 if hi < 20000 else lo * 0.55, y, sub, ha="left" if hi < 20000 else "right", va="center", fontsize=5.4, color=C_MUTED)

    band_bar(2.4, near_km[0] + 30, max(near_km[1], 80), C_OK, "NEAR", "mesoscale coherence\n(relative to graph)")
    band_bar(1.5, mid_km[0], mid_km[1], C_WARN, "MID", "frontal / basin contrast\n(gain suppressed)")
    band_bar(0.6, far_km[0], min(far_km[1], 35000), C_ETR, "FAR", "shared-mode uplift")

    ax.axvline(rossby, color=C_SALTON, lw=1.2, ls="--")
    ax.scatter([rossby], [3.05], marker="v", color=C_SALTON, s=40, zorder=5)
    ax.text(rossby, 3.05, "  ~50 km Rossby\n  (not claimed)", fontsize=5.6, color=C_SALTON, va="top", ha="left")
    ax.axvline(phys["km_at_g_min"], color=C_MUTED, lw=0.9, ls=":")
    ax.text(
        phys["km_at_g_min"],
        -0.05,
        rf"$g_{{\min}}$ ~{phys['km_at_g_min']:.0f} km",
        fontsize=5.4,
        color=C_MUTED,
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        0.99,
        0.08,
        "relative to graph-diffusion on 128 sensors — not a new ocean law",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.4,
        color=C_MUTED,
        style="italic",
    )

    save_fig(fig, out_dir, "edfig6_sstphys", dpi)


def main():
    args = parse_args()
    out_dir = resolve(args.out_dir)
    only = {s.strip() for s in args.only.split(",")}
    if "all" in only:
        only = {"1", "2", "3", "4", "5", "6"}

    dispatch = {
        "1": plot_ed1,
        "2": plot_ed2,
        "3": plot_ed3,
        "4": plot_ed4,
        "5": plot_ed5,
        "6": plot_ed6,
    }
    for key in sorted(only):
        if key not in dispatch:
            raise SystemExit(f"unknown ED id: {key}")
        dispatch[key](out_dir, args.dpi)


if __name__ == "__main__":
    main()
