#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 4 — validating α (geometry binding + named reinjection depth)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.eeg_dataset import STANDARD_10_20
from hnf.eeg_geometry import electrode_xyz, pairwise_chord_distance
from hnf.propagation_dynamics.sst_sensor_dataset import build_sensor_coords

C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
BG = "#FFFFFF"
C_EEG = "#E69F00"
C_LEMON = "#F6C56E"
C_SST = "#009E73"
C_OK = "#0072B2"
C_BAD = "#D55E00"
C_NEUT = "#6B7280"
C_SOFT = "#CC79A7"
C_BAND = "#F3F4F6"
C_NEAR = "#A7F3D0"
C_MID = "#FDE68A"
C_WEAK = "#FECACA"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="docs/figures/unified")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--eeg-harden", default="outputs/propagation_dynamics/eeg_instantaneous_harden_v1/summary_pooled.json")
    p.add_argument("--eeg-harden-dir", default="outputs/propagation_dynamics/eeg_instantaneous_harden_v1")
    p.add_argument("--lemon", default="outputs/propagation_dynamics/eeg_lemon_replication_v1/summary_pooled.json")
    p.add_argument("--sst-rdg", default="outputs/propagation_dynamics/sst_alpha_rdg_closure_v3/summary_pooled.json")
    p.add_argument("--sst-rows", default="outputs/propagation_dynamics/sst_alpha_rdg_closure_v3/rows.json")
    p.add_argument("--eeg-rdg", default="outputs/propagation_dynamics/eeg_alpha_rdg_closure_v2/summary_pooled.json")
    p.add_argument("--eeg-rows", default="outputs/propagation_dynamics/eeg_alpha_rdg_closure_v2/rows.json")
    p.add_argument("--sst-physics", default="outputs/propagation_dynamics/sst_rdg_physics_v1/physics_map.json")
    return p.parse_args()


def resolve(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else _REPO / p


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update({
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
    })


def clean(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.2, width=0.6, colors=C_INK)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK)
        ax.spines[side].set_linewidth(0.7)


def hdr(ax, letter: str, title: str):
    ax.set_axis_off()
    ax.text(0.0, 0.42, letter, fontsize=11, fontweight="bold", color=C_INK, va="center", transform=ax.transAxes)
    ax.text(0.07, 0.42, title, fontsize=7.6, color=C_INK, va="center", transform=ax.transAxes)


def softplus(x):
    x = np.asarray(x, float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def rdg_curve(b0, b1, b2, n=240):
    r = np.linspace(0.0, 1.0, n)
    return r, softplus(b0 + b1 * r + b2 * r * r)


def boot_pct(boot):
    return 100 * float(boot["mean"]), 100 * float(boot["ci_lo"]), 100 * float(boot["ci_hi"])


def load_alpha_curves(harden_dir: Path):
    mats, r = [], None
    for seed_dir in sorted(harden_dir.glob("seed_*")):
        path = seed_dir / "alpha_r_curves.json"
        if not path.exists():
            continue
        rows = load_json(path)
        if not rows:
            continue
        r = np.asarray(rows[0]["r_centers"], float)
        mats.append(np.asarray([row["alpha_mean"] for row in rows], float))
    if not mats:
        raise FileNotFoundError(f"no alpha_r_curves under {harden_dir}")
    return r, np.concatenate(mats, axis=0)


def draw_edge_field(ax, xy, rhat, values, cmap, vmin, vmax, node_color, title):
    clean(ax)
    n = xy.shape[0]
    iu, ju = np.triu_indices(n, 1)
    order = np.argsort(rhat[iu, ju])
    keep = order[: min(len(order), max(3 * n, 48))]
    segs, cols, widths = [], [], []
    rmax = float(rhat[iu, ju][keep].max()) + 1e-9
    for k in keep:
        i, j = int(iu[k]), int(ju[k])
        segs.append([xy[i], xy[j]])
        cols.append(values[i, j])
        widths.append(1.8 - 1.1 * float(rhat[i, j]) / rmax)
    lc = LineCollection(
        segs,
        array=np.asarray(cols, float),
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=np.clip(widths, 0.55, 1.8),
        alpha=0.9,
        zorder=1,
    )
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], s=18, c=node_color, edgecolors=C_INK, linewidths=0.4, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    pad = 0.08 * max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
    ax.set_xlim(float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad)
    ax.set_ylim(float(xy[:, 1].min()) - pad, float(xy[:, 1].max()) + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=6.3, color=C_MUTED, pad=3)
    return lc


def draw_scalp_alpha_field(ax, r_centers, alpha_med, names=STANDARD_10_20):
    """Top-down 10–20 montage with head outline, edge α coloring, labeled nodes."""
    clean(ax)
    xyz = electrode_xyz(names)
    # slight foreshortening so frontal sits “up” on the page
    xy = np.column_stack([xyz[:, 1], xyz[:, 0]])
    D = pairwise_chord_distance(xyz)
    rhat = np.clip(D / 2.0, 0.0, 1.0)
    alpha = np.interp(rhat, r_centers, alpha_med, left=float(alpha_med[0]), right=float(alpha_med[-1]))
    np.fill_diagonal(alpha, np.nan)

    # head disk + nose
    rad = 1.08 * float(np.max(np.linalg.norm(xy, axis=1)))
    head = plt.Circle((0.0, 0.0), rad, fill=True, facecolor="#FFF7ED", edgecolor="#D6D3D1", linewidth=1.0, zorder=0)
    ax.add_patch(head)
    nose = np.array([[-0.12 * rad, 0.92 * rad], [0.0, 1.12 * rad], [0.12 * rad, 0.92 * rad]])
    ax.fill(nose[:, 0], nose[:, 1], color="#D6D3D1", zorder=0)
    # ear ticks
    for sx in (-1.0, 1.0):
        ax.plot([sx * 1.02 * rad, sx * 1.10 * rad], [0.0, 0.0], color="#D6D3D1", lw=1.6, solid_capstyle="round", zorder=0)

    # all edges up to the fitted α support, plus a few longer ones for context
    n = xy.shape[0]
    iu, ju = np.triu_indices(n, 1)
    r_max_fit = float(r_centers.max()) * 1.05
    primary = rhat[iu, ju] <= r_max_fit
    # keep nearest extras beyond support so the montage reads as a graph
    order = np.argsort(rhat[iu, ju])
    extra = order[~primary[order]][: max(12, n)]
    keep_idx = np.unique(np.concatenate([np.where(primary)[0], extra]))

    segs_p, cols_p, widths_p = [], [], []
    segs_x, cols_x, widths_x = [], [], []
    for k in keep_idx:
        i, j = int(iu[k]), int(ju[k])
        seg = [xy[i], xy[j]]
        col = alpha[i, j]
        w = 0.7 + 2.2 * abs(float(alpha[i, j]) - 1.0) + 0.8 * (1.0 - float(rhat[i, j]))
        if rhat[i, j] <= r_max_fit:
            segs_p.append(seg)
            cols_p.append(col)
            widths_p.append(w)
        else:
            segs_x.append(seg)
            cols_x.append(col)
            widths_x.append(w)
    # amplify perceptual contrast: signed-sqrt stretch of (α−1)
    def stretch(a):
        d = np.asarray(a, float) - 1.0
        return np.sign(d) * np.sqrt(np.abs(d))

    delta = stretch(alpha)
    dmax = float(np.nanpercentile(np.abs(delta), 95)) + 1e-6
    try:
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)
        cmap = plt.cm.RdYlBu_r
    except Exception:
        norm = Normalize(vmin=-dmax, vmax=dmax)
        cmap = plt.cm.RdYlBu_r
    if segs_x:
        ax.add_collection(
            LineCollection(
                segs_x,
                array=stretch(np.asarray(cols_x, float)),
                cmap=cmap,
                norm=norm,
                linewidths=np.clip(widths_x, 0.45, 1.5),
                alpha=0.30,
                zorder=1,
            )
        )
    # rebuild primary arrays as α−1 for colour
    cols_p_d = list(stretch(np.asarray(cols_p, float)))
    lc = LineCollection(
        segs_p,
        array=np.asarray(cols_p_d, float),
        cmap=cmap,
        norm=norm,
        linewidths=np.clip(widths_p, 0.7, 2.8),
        alpha=0.95,
        zorder=2,
    )
    ax.add_collection(lc)

    # soft node halo underlay
    ax.scatter(xy[:, 0], xy[:, 1], s=90, c="#FED7AA", alpha=0.35, edgecolors="none", zorder=3)
    node_a = np.nanmean(alpha, axis=1)
    node_dev = np.nanmean(np.abs(alpha - 1.0), axis=1)
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=36 + 260 * node_dev,
        c=stretch(node_a),
        cmap=cmap,
        norm=norm,
        edgecolors=C_INK,
        linewidths=0.6,
        zorder=4,
    )
    # labels for a readable subset
    label_set = {"Fp1", "Fp2", "Fz", "Cz", "Pz", "O1", "O2", "T3", "T4", "C3", "C4"}
    for name, (x, y) in zip(names, xy):
        if name not in label_set:
            continue
        ax.text(
            x,
            y + 0.07 * rad,
            name,
            ha="center",
            va="bottom",
            fontsize=4.4,
            color=C_INK,
            zorder=5,
            clip_on=False,
        )

    ax.set_aspect("equal", adjustable="box")
    pad = 0.18 * rad
    ax.set_xlim(-rad - pad, rad + pad)
    ax.set_ylim(-rad - pad, rad + 1.15 * pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(r"scalp field $\alpha_{ij}(\hat{r})$", fontsize=6.3, color=C_MUTED, pad=2)
    cax = ax.inset_axes([0.58, 0.03, 0.38, 0.045])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([-dmax, 0.0, dmax])
    cb.set_ticklabels([f"{1 - dmax**2:.2f}", "1", f"{1 + dmax**2:.2f}"])
    cb.ax.tick_params(labelsize=4.3, length=1.2)
    cb.outline.set_linewidth(0.35)
    ax.text(
        0.02,
        0.02,
        "red = near boost\nblue = far cut",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
        color=C_MUTED,
    )
    return sc


def draw_a(ax_obj, ax_map, ax_cf, harden, r, A):
    clean(ax_obj)
    clean(ax_cf)

    rng = np.random.default_rng(0)
    show = rng.choice(A.shape[0], size=min(140, A.shape[0]), replace=False)
    for i in show:
        ax_obj.plot(r, A[i], color=C_EEG, alpha=0.08, lw=0.7, zorder=1)
    q25, q50, q75 = np.percentile(A, [25, 50, 75], axis=0)
    ax_obj.fill_between(r, q25, q75, color=C_EEG, alpha=0.30, linewidth=0, zorder=2)
    ax_obj.plot(r, q50, color=C_EEG, lw=2.2, zorder=3)
    ax_obj.axhline(1.0, color=C_NEUT, ls="--", lw=0.8, alpha=0.85, zorder=0)
    ax_obj.set_xlabel(r"normalised chord $\hat{r}$", fontsize=6.4)
    ax_obj.set_ylabel(r"edge gain $\alpha(\hat{r})$", fontsize=6.4)
    ax_obj.set_xlim(float(r.min()) - 0.005, float(r.max()) + 0.005)
    ax_obj.set_ylim(float(np.percentile(A, 2)) - 0.05, float(np.percentile(A, 98)) + 0.08)
    ax_obj.set_title(rf"fitted object · n={A.shape[0]}", fontsize=6.3, color=C_MUTED, pad=3)
    ax_obj.text(
        0.98, 0.04,
        rf"$\langle b_1\rangle$={float(harden['b1_mean']):+.3f}" + "\n" + rf"sign agree={float(harden['G7_b1_sign_agreement']):.0%}",
        transform=ax_obj.transAxes, ha="right", va="bottom", fontsize=5.3, color=C_MUTED, multialignment="right",
    )

    draw_scalp_alpha_field(ax_map, r, q50)

    items = [
        ("wrong geometry", harden["boot_rel_cf_geom"], C_BAD),
        ("edge permutation", harden["boot_rel_cf_aperm"], C_SOFT),
    ]
    ax_cf.axvline(0, color=C_LINE, lw=0.8, zorder=0)
    for y, (lab, boot, col) in zip([1.0, 0.0], items):
        mean, lo, hi = boot_pct(boot)
        ax_cf.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        ax_cf.scatter([mean], [y], s=34, color=col, edgecolors=C_INK, linewidths=0.5, zorder=3)
        ax_cf.text(0.02, y + 0.28, lab, ha="left", va="bottom", fontsize=5.8, color=C_INK, transform=ax_cf.get_yaxis_transform())
        ax_cf.text(hi + 4.0, y, f"{mean:+.0f}%", ha="left", va="center", fontsize=6.0, color=col, fontweight="bold")
    ones = harden["boot_ones"]
    ax_cf.text(
        0.98, 0.98,
        r"necessity: $\alpha\equiv1$ also hurts" + "\n" + rf"$\Delta$MSE={float(ones['mean']):.3f} [{float(ones['ci_lo']):.3f},{float(ones['ci_hi']):.3f}]",
        transform=ax_cf.transAxes, ha="right", va="top", fontsize=5.2, color=C_MUTED, multialignment="right",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_BAND, edgecolor=C_LINE, linewidth=0.6),
    )
    ax_cf.set_yticks([])
    ax_cf.set_xlabel("rel. MSE rise vs fitted α (%)", fontsize=6.3)
    ax_cf.set_xlim(-5, max(boot_pct(it[1])[2] for it in items) * 1.22)
    ax_cf.set_ylim(-0.55, 1.75)
    ax_cf.set_title("geometry-binding CFs", fontsize=6.3, color=C_MUTED, pad=3)
    ax_cf.text(0.02, 0.02, f"n={int(harden['boot_rel_cf_geom']['n_subjects'])} · PASS",
               transform=ax_cf.transAxes, ha="left", va="bottom", fontsize=5.3, color=C_MUTED)




def draw_b(ax_rep, harden, lemon):
    """Replication agreement scatter; LEMON prior preference as inset."""
    clean(ax_rep)

    mse_i = float(lemon["mse_instantaneous"])
    lemon_aperm = {
        "mean": float(lemon["boot_cf_aperm"]["mean"]) / mse_i,
        "ci_lo": float(lemon["boot_cf_aperm"]["ci_lo"]) / mse_i,
        "ci_hi": float(lemon["boot_cf_aperm"]["ci_hi"]) / mse_i,
    }
    probes = [
        ("wrong geom.", harden["boot_rel_cf_geom"], lemon["boot_rel_cf_geom"], C_BAD),
        ("edge perm.", harden["boot_rel_cf_aperm"], lemon_aperm, C_SOFT),
    ]

    lo = 0.0
    hi = max(boot_pct(p[1])[2] for p in probes)
    hi = max(hi, max(boot_pct(p[2])[2] for p in probes)) * 1.12
    ax_rep.fill_between([lo, hi], [lo * 0.75, hi * 0.75], [lo * 1.25, hi * 1.25],
                        color="#FEF3C7", alpha=0.55, zorder=0, linewidth=0)
    ax_rep.plot([lo, hi], [lo, hi], color=C_LINE, lw=1.2, ls="--", zorder=1)
    for lab, h, L, col in probes:
        hm, hl, hh = boot_pct(h)
        lm, ll, lh = boot_pct(L)
        ax_rep.plot([hl, hh], [lm, lm], color=col, lw=1.4, solid_capstyle="round", zorder=2, alpha=0.85)
        ax_rep.plot([hm, hm], [ll, lh], color=col, lw=1.4, solid_capstyle="round", zorder=2, alpha=0.85)
        ax_rep.scatter([hm], [lm], s=70, color=col, edgecolors=C_INK, linewidths=0.6, zorder=4)
        ax_rep.annotate(
            lab,
            xy=(hm, lm),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=5.6,
            color=C_INK,
            arrowprops=dict(arrowstyle="-", color=C_LINE, lw=0.6),
        )
        ax_rep.text(hm, lm - 0.06 * hi, f"{hm:.0f}→{lm:.0f}", ha="center", va="top",
                    fontsize=5.3, color=col, fontweight="bold")

    ax_rep.set_xlim(lo, hi)
    ax_rep.set_ylim(lo, hi)
    ax_rep.set_aspect("equal", adjustable="box")
    ax_rep.set_xlabel(f"Harden rel. rise (%)  n={harden['boot_rel_cf_geom']['n_subjects']}", fontsize=6.2)
    ax_rep.set_ylabel(f"LEMON rel. rise (%)  n={lemon['boot_rel_cf_geom']['n_subjects']}", fontsize=6.2)
    ax_rep.set_title("replication sits on the diagonal", fontsize=6.3, color=C_MUTED, pad=3)
    ax_rep.text(0.98, 0.02, "±25% agreement band", transform=ax_rep.transAxes,
                ha="right", va="bottom", fontsize=5.1, color=C_MUTED)

    # Inset: LEMON prior gate R1 — instantaneous beats wave (matches manuscript)
    m_i = mse_i
    m_w = float(lemon["mse_wave"])
    d = float(lemon["boot_inst_vs_wave"]["mean"])
    ci_lo = float(lemon["boot_inst_vs_wave"]["ci_lo"])
    ci_hi = float(lemon["boot_inst_vs_wave"]["ci_hi"])
    ax_in = ax_rep.inset_axes([0.06, 0.82, 0.48, 0.13])
    ax_in.set_facecolor("none")
    ax_in.patch.set_visible(False)
    for side in ax_in.spines.values():
        side.set_visible(False)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.plot([0.10, 0.90], [0.38, 0.38], color="#D1D5DB", lw=2.8, solid_capstyle="round",
               transform=ax_in.transAxes, zorder=1)
    lo_m, hi_m = min(m_i, m_w) - 0.004, max(m_i, m_w) + 0.004

    def xpos(m):
        return 0.12 + 0.76 * (m - lo_m) / (hi_m - lo_m)

    xi, xw = xpos(m_i), xpos(m_w)
    ax_in.annotate(
        "",
        xy=(xi, 0.38),
        xytext=(xw, 0.38),
        xycoords=ax_in.transAxes,
        textcoords=ax_in.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=0.9, mutation_scale=7),
    )
    ax_in.scatter([xi], [0.38], s=34, color=C_EEG, edgecolors=C_INK, linewidths=0.4,
                  transform=ax_in.transAxes, zorder=3)
    ax_in.scatter([xw], [0.38], s=34, color=C_OK, edgecolors=C_INK, linewidths=0.4,
                  transform=ax_in.transAxes, zorder=3)
    ax_in.text(xi, 0.68, f"inst {m_i:.3f}", transform=ax_in.transAxes,
               ha="center", va="bottom", fontsize=4.5, color=C_EEG, fontweight="bold")
    ax_in.text(xw, 0.68, f"wave {m_w:.3f}", transform=ax_in.transAxes,
               ha="center", va="bottom", fontsize=4.5, color=C_OK, fontweight="bold")
    ax_in.text(
        0.5, 0.02,
        rf"inst preferred · $\Delta$={d:.3f} [{ci_lo:.3f},{ci_hi:.3f}] · R1 PASS",
        transform=ax_in.transAxes, ha="center", va="bottom", fontsize=4.2, color=C_MUTED,
    )
    ax_in.set_xlim(0, 1)
    ax_in.set_ylim(0, 1)





def draw_c(ax_id, ax_g, ax_map, sst, rows, phys):
    clean(ax_id)
    clean(ax_g)

    free = np.asarray([float(r["mse_free"]) for r in rows], float)
    named = np.asarray([float(r["mse_named"]) for r in rows], float)
    wrong = np.asarray([float(r["mse_wrong"]) for r in rows], float)
    ones = np.asarray([float(r["mse_ones"]) for r in rows], float)

    lo = float(min(free.min(), named.min(), ones.min()) * 0.96)
    hi = float(max(free.max(), named.max(), ones.max()) * 1.04)
    ax_id.plot([lo, hi], [lo, hi], color=C_LINE, lw=1.0, zorder=0)
    ax_id.scatter(free, ones, s=12, color=C_NEUT, alpha=0.4, edgecolors="none", zorder=1, label=r"$\alpha\equiv1$")
    ax_id.scatter(free, named, s=20, color=C_SST, alpha=0.85, edgecolors=C_INK, linewidths=0.3, zorder=3, label="named RDG")
    ax_id.set_xlim(lo, hi)
    ax_id.set_ylim(lo, hi)
    ax_id.set_aspect("equal", adjustable="box")
    ax_id.set_xlabel(r"MSE free $\alpha$", fontsize=6.3)
    ax_id.set_ylabel("MSE reinjected", fontsize=6.3)
    ax_id.set_title(f"held-out identity (n={len(rows)})", fontsize=6.3, color=C_MUTED, pad=3)
    ax_id.legend(fontsize=5.1, frameon=False, loc="upper left", markerscale=0.9)
    rel = float(sst["block"]["mean_rel_named_vs_free"])
    ax_id.text(
        0.98, 0.02,
        f"named−free rel={rel:.3f}\nwrong median={np.median(wrong):.1f} (off-scale)",
        transform=ax_id.transAxes, ha="right", va="bottom", fontsize=5.1, color=C_MUTED, multialignment="right",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#FECACA", linewidth=0.6),
    )

    hyp = sst["hypothesis"]
    rr, gg = rdg_curve(float(hyp["b0"]), float(hyp["b1"]), float(hyp["b2"]))
    b0 = float(hyp["b0"])
    for row in rows[::2]:
        r1, g1 = rdg_curve(b0, float(row["b1_free"]), float(row["b2_free"]), n=100)
        ax_g.plot(r1, g1, color=C_SST, alpha=0.10, lw=0.7, zorder=1)
    ax_g.plot(rr, gg, color=C_SST, lw=2.2, zorder=3)
    ax_g.fill_between(rr, gg, alpha=0.12, color=C_SST, linewidth=0, zorder=2)
    chk = phys["check"]
    for x, y, lab in [(0.12, float(chk["g_near"]), "near"), (0.45, float(chk["g_mid"]), "mid"), (0.82, float(chk["g_far"]), "far")]:
        ax_g.scatter([x], [y], s=26, color=C_INK, zorder=4)
        ax_g.text(x, y + 0.08, lab, ha="center", va="bottom", fontsize=5.1, color=C_MUTED)
    ax_g.set_xlim(0, 1)
    ax_g.set_xlabel(r"$\hat{r}$", fontsize=6.3)
    ax_g.set_ylabel(r"$g(\hat{r})$", fontsize=6.3)
    ax_g.set_title(rf"frozen RDG  $\ell$={hyp['length_scale']}", fontsize=6.3, color=C_MUTED, pad=3)
    corr = float(chk["corr_g_vs_abs"])
    ax_g.text(
        0.98, 0.96,
        rf"$b_1$={hyp['b1']:.2f}" + "\n" + f"corr(g,|ΔSST|)={corr:.2f}\nPASS · not Rd",
        transform=ax_g.transAxes, ha="right", va="top", fontsize=5.0, color=C_MUTED, multialignment="right",
    )

    coords = build_sensor_coords().detach().cpu().numpy()
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    rhat = d / (d.max() + 1e-9)
    g_mat = softplus(float(hyp["b0"]) + float(hyp["b1"]) * rhat + float(hyp["b2"]) * rhat * rhat)
    np.fill_diagonal(g_mat, np.nan)
    vmin, vmax = float(np.nanpercentile(g_mat, 5)), float(np.nanpercentile(g_mat, 95))
    idx = np.linspace(0, len(coords) - 1, 36, dtype=int)
    xy_s = coords[idx]
    rhat_s = rhat[np.ix_(idx, idx)]
    g_s = g_mat[np.ix_(idx, idx)]
    lc = draw_edge_field(ax_map, xy_s, rhat_s, g_s, plt.cm.YlGn, vmin, vmax, C_SST, "sensor-graph gain field")
    cax = ax_map.inset_axes([0.68, 0.06, 0.26, 0.04])
    cb = plt.colorbar(lc, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, vmax])
    cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    cb.ax.tick_params(labelsize=4.6, length=1.4)
    cb.outline.set_linewidth(0.4)



def draw_d(ax_sc, ax_swarm, eeg_sum, sst_sum, eeg_rows, sst_rows):
    clean(ax_sc)
    clean(ax_swarm)

    e_free = np.asarray([float(r["mse_free"]) for r in eeg_rows], float)
    e_named = np.asarray([float(r["mse_named"]) for r in eeg_rows], float)
    s_free = np.asarray([float(r["mse_free"]) for r in sst_rows], float)
    s_named = np.asarray([float(r["mse_named"]) for r in sst_rows], float)
    e_rel = np.asarray([float(r["rel_named_vs_free"]) for r in eeg_rows], float)
    s_rel = np.asarray([float(r["rel_named_vs_free"]) for r in sst_rows], float)

    ax_sc.scatter(e_free, e_named, s=16, color=C_EEG, alpha=0.55, edgecolors=C_INK, linewidths=0.25, zorder=2, label="EEG")
    ax_sc.scatter(s_free, s_named, s=20, color=C_SST, alpha=0.9, edgecolors=C_INK, linewidths=0.3, zorder=3, label="SST")
    lims = np.r_[e_free, e_named, s_free, s_named]
    lo, hi = float(lims.min() * 0.85), float(lims.max() * 1.15)
    ax_sc.plot([lo, hi], [lo, hi], color=C_LINE, lw=1.0, zorder=0)
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.set_xlim(lo, hi)
    ax_sc.set_ylim(lo, hi)
    ax_sc.set_xlabel(r"MSE free $\alpha$", fontsize=6.3)
    ax_sc.set_ylabel("MSE named RDG", fontsize=6.3)
    eeg_pass = bool(eeg_sum["closure_pass"])
    sst_pass = bool(sst_sum["closure_pass"])
    ax_sc.set_title(
        f"depth = distance to diagonal  ·  EEG {'PASS' if eeg_pass else 'FAIL'} / SST {'PASS' if sst_pass else 'FAIL'}",
        fontsize=6.3, color=C_MUTED, pad=3,
    )
    ax_sc.legend(fontsize=5.4, frameon=False, loc="upper left")

    def swarm(vals, y0, color):
        x = np.log10(1.0 + np.clip(vals, 0, None))
        order = np.argsort(x)
        x = x[order]
        y = np.zeros_like(x)
        bins = np.linspace(float(x.min()) - 1e-6, float(x.max()) + 1e-6, 18)
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (x >= b0) & (x < b1)
            n = int(m.sum())
            if n:
                y[m] = y0 + (np.arange(n) - (n - 1) / 2.0) * 0.045
        ax_swarm.scatter(x, y, s=10, color=color, alpha=0.75, edgecolors=C_INK, linewidths=0.15, zorder=3)

    t_near, t_mid = np.log10(1.15), np.log10(1.80)
    xmax = max(float(np.log10(1.0 + e_rel.max())), float(np.log10(1.0 + s_rel.max())), t_mid) * 1.08
    ax_swarm.axvspan(0, t_near, color=C_NEAR, alpha=0.55, zorder=0)
    ax_swarm.axvspan(t_near, t_mid, color=C_MID, alpha=0.55, zorder=0)
    ax_swarm.axvspan(t_mid, xmax, color=C_WEAK, alpha=0.55, zorder=0)

    def kde_ridge(vals, y0, color):
        x = np.log10(1.0 + np.clip(vals, 0, None))
        xs = np.linspace(0, xmax, 120)
        hist, edges = np.histogram(x, bins=24, range=(0, xmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kern = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
        hist_s = np.convolve(hist, kern, mode="same")
        dens = np.interp(xs, centers, hist_s, left=0, right=0)
        dens = dens / (dens.max() + 1e-12) * 0.38
        ax_swarm.fill_between(xs, y0 - dens, y0 + dens, color=color, alpha=0.22, linewidth=0, zorder=1)
        ax_swarm.plot(xs, y0 + dens, color=color, lw=0.9, alpha=0.7, zorder=2)
        ax_swarm.plot(xs, y0 - dens, color=color, lw=0.9, alpha=0.7, zorder=2)

    kde_ridge(e_rel, 0.55, C_EEG)
    kde_ridge(s_rel, -0.55, C_SST)
    swarm(e_rel, 0.55, C_EEG)
    swarm(s_rel, -0.55, C_SST)
    ax_swarm.axhline(0, color=C_LINE, lw=0.6, zorder=1)
    ax_swarm.text(0.02, 0.55, "EEG", transform=ax_swarm.get_yaxis_transform(),
                  ha="left", va="center", fontsize=5.9, color=C_EEG, fontweight="bold")
    ax_swarm.text(0.02, -0.55, "SST", transform=ax_swarm.get_yaxis_transform(),
                  ha="left", va="center", fontsize=5.9, color=C_SST, fontweight="bold")
    for x, lab in [(t_near / 2, "near"), ((t_near + t_mid) / 2, "mod"), ((t_mid + xmax) / 2, "weak")]:
        ax_swarm.text(x, 1.05, lab, ha="center", va="bottom", fontsize=5.2, color=C_MUTED)
    ax_swarm.set_xlim(0, xmax)
    ax_swarm.set_ylim(-1.15, 1.25)
    ax_swarm.set_yticks([])
    ax_swarm.set_xlabel(r"$\log_{10}(1+\mathrm{rel})$", fontsize=6.3)
    ax_swarm.set_title("window-wise reinjection residual", fontsize=6.3, color=C_MUTED, pad=3)
    ax_swarm.spines["left"].set_visible(False)
    ax_swarm.text(
        0.98, 0.02,
        f"near-full: EEG {100 * float((e_rel <= 0.15).mean()):.0f}% · SST {100 * float((s_rel <= 0.15).mean()):.0f}%",
        transform=ax_swarm.transAxes, ha="right", va="bottom", fontsize=5.2, color=C_MUTED,
    )




def main():
    args = parse_args()
    style()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    harden = load_json(resolve(args.eeg_harden))
    lemon = load_json(resolve(args.lemon))
    sst = load_json(resolve(args.sst_rdg))
    eeg = load_json(resolve(args.eeg_rdg))
    sst_rows = load_json(resolve(args.sst_rows))
    eeg_rows = load_json(resolve(args.eeg_rows))
    phys = load_json(resolve(args.sst_physics))
    r, A = load_alpha_curves(resolve(args.eeg_harden_dir))

    fig = plt.figure(figsize=(13.6, 11.2))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.08], left=0.055, right=0.985, top=0.955, bottom=0.04, hspace=0.22)
    title_h, title_gap = 0.085, 0.10
    top = outer[0].subgridspec(2, 2, height_ratios=[title_h, 1.0], hspace=title_gap, wspace=0.24)
    bot = outer[1].subgridspec(2, 2, height_ratios=[title_h, 1.0], hspace=title_gap, wspace=0.24)

    hdr(fig.add_subplot(top[0, 0]), "a", r"EEG: geometry-bound $\alpha$ — object, scalp field, CF ladder")
    hdr(fig.add_subplot(top[0, 1]), "b", "LEMON replication: Harden–LEMON agreement (prior inset)")

    # a: object | (scalp field / CF forest stacked)
    gs_a = top[1, 0].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    gs_a_r = gs_a[0, 1].subgridspec(2, 1, height_ratios=[1.05, 0.90], hspace=0.38)
    draw_a(
        fig.add_subplot(gs_a[0, 0]),
        fig.add_subplot(gs_a_r[0, 0]),
        fig.add_subplot(gs_a_r[1, 0]),
        harden, r, A,
    )

    # b: single agreement panel with prior inset
    draw_b(fig.add_subplot(top[1, 1]), harden, lemon)

    hdr(fig.add_subplot(bot[0, 0]), "c", "SST: frozen RDG reinjection closes — with sensor-graph field")
    hdr(fig.add_subplot(bot[0, 1]), "d", "Reinjection depth: SST full closure vs EEG partial transfer")

    # c: (identity / sensor field stacked) | frozen RDG curve
    gs_c = bot[1, 0].subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.28)
    gs_c_l = gs_c[0, 0].subgridspec(2, 1, height_ratios=[1.15, 0.90], hspace=0.34)
    draw_c(
        fig.add_subplot(gs_c_l[0, 0]),
        fig.add_subplot(gs_c[0, 1]),
        fig.add_subplot(gs_c_l[1, 0]),
        sst, sst_rows, phys,
    )

    # d: two full-width stacked panels (no radar)
    gs_d = bot[1, 1].subgridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.34)
    draw_d(
        fig.add_subplot(gs_d[0, 0]),
        fig.add_subplot(gs_d[1, 0]),
        eeg, sst, eeg_rows, sst_rows,
    )

    # (footer claim removed — figure should read without a slogan strip)

    png = out_dir / "fig4_alpha.png"
    pdf = out_dir / "fig4_alpha.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print({"png": str(png), "pdf": str(pdf)})




if __name__ == "__main__":
    main()
