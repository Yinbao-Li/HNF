#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Impact-discovery plate for EEG diffusion / Huygens leftover claim.

Style locked to seismic journal plates (plot_shape_geo_coda_figure /
plot_nc_main_figures): bold panel letters, 3D boom pies, case waveforms,
no text overflow / occlusion. Public vocabulary only (no internal model codes).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import Rbf

from hnf.eeg_geometry import electrode_xyz
from hnf.eeg_subject_diffusion import residualize, sex_to_float, spearman_r

BG = "#FFFFFF"
C_INK = "#111827"
C_MUTED = "#6B7280"
C_LINE = "#E5E7EB"
C_TEAL = "#0F766E"
C_GOLD = "#B45309"
C_ABS = "#B91C1C"
C_RING = "#1D4ED8"
C_HC = "#1B6B93"
C_FTD = "#C45C26"
C_AD = "#3D5A5B"
C_YOUNG = "#0F766E"
C_OLD = "#B45309"
GROUP_COLORS = {"HC": C_HC, "FTD": C_FTD, "AD": C_AD}

OUT = _REPO / "docs/figures/eeg"
CACHE = _REPO / "outputs/eeg/probe_publishable/impact_case_cache.npz"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BG,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "text.color": C_INK,
        }
    )


def _panel_header(ax, letter: str, title: str, y: float = 1.02, title_dx_pt: float = 16.0) -> None:
    """Bold panel letter + title offset in points (no overlap with letter)."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight=700,
        fontfamily="DejaVu Sans",
        color="#000000",
        va="bottom",
        ha="left",
        zorder=30,
        clip_on=False,
    )
    ax.annotate(
        title,
        xy=(0.0, y),
        xycoords=ax.transAxes,
        xytext=(title_dx_pt, 0.0),
        textcoords="offset points",
        fontsize=9,
        color=C_INK,
        va="bottom",
        ha="left",
        zorder=30,
        clip_on=False,
    )


def _clean(ax) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, colors=C_INK)


def _box(ax, x, y, w, h, fc, ec="#D1D5DB", lw=0.85, r=0.08, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=z,
            clip_on=False,
        )
    )


def _fit_text(ax, x, y, text, w, h, fontsize=7.2, **kw):
    """Center text; shrink fontsize until estimated width fits box."""
    fs = fontsize
    for _ in range(6):
        # rough char-width heuristic in data units
        est = 0.55 * fs / 8.0 * max(len(line) for line in text.split("\n"))
        if est <= w * 0.92:
            break
        fs -= 0.5
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C_INK, **kw)


# ---------------------------------------------------------------------------
# 3D boom pie
# ---------------------------------------------------------------------------


def _boom_pie_3d(fig, host_ax, labels, sizes, colors, total_label: str) -> None:
    host_ax.set_xlim(0, 1)
    host_ax.set_ylim(0, 1)
    host_ax.axis("off")
    host_ax.set_facecolor(BG)

    sizes = np.asarray(sizes, dtype=float)
    total = float(sizes.sum())
    fracs = sizes / max(total, 1.0)

    fig.canvas.draw()
    bbox = host_ax.get_position()
    leg_h = 0.18 * bbox.height
    bot_h = 0.08 * bbox.height
    avail_h = bbox.height - leg_h - bot_h
    side = min(1.08 * avail_h, 1.40 * bbox.width)
    x0 = bbox.x0 + 0.5 * (bbox.width - side)
    # pin pie near bottom of host so unused height sits under the legend, not above row 2
    y0 = bbox.y0 + bot_h * 0.35
    ax3d = fig.add_axes([x0, y0, side, side], projection="3d", facecolor=BG)
    ax3d.set_zorder(3)
    ax3d.set_axis_off()
    ax3d.set_box_aspect((1.0, 1.0, 0.42))
    ax3d.view_init(elev=28, azim=-50)
    ax3d.set_xlim(-1.45, 1.45)
    ax3d.set_ylim(-1.45, 1.45)
    ax3d.set_zlim(0, 0.55)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BG)
    ax3d.grid(False)

    height = 0.34
    explode = 0.16
    theta0 = 0.5 * np.pi
    for frac, col in zip(fracs, colors):
        dtheta = 2.0 * np.pi * float(frac)
        nseg = max(18, int(72 * frac) + 2)
        thetas = np.linspace(theta0, theta0 - dtheta, nseg)
        theta_mid = theta0 - 0.5 * dtheta
        ox = explode * np.cos(theta_mid)
        oy = explode * np.sin(theta_mid)
        rgb = np.array(to_rgb(col))
        top_rgb = np.clip(rgb * 1.08 + 0.05, 0, 1)
        side_rgb = np.clip(rgb * 0.68, 0, 1)
        xt = ox + np.concatenate([[0.0], np.cos(thetas), [0.0]])
        yt = oy + np.concatenate([[0.0], np.sin(thetas), [0.0]])
        zt = np.full_like(xt, height)
        ax3d.add_collection3d(
            Poly3DCollection(
                [list(zip(xt, yt, zt))],
                facecolors=[top_rgb],
                edgecolors="white",
                linewidths=0.4,
                alpha=1.0,
            )
        )
        for i in range(len(thetas) - 1):
            xa, ya = ox + np.cos(thetas[i]), oy + np.sin(thetas[i])
            xb, yb = ox + np.cos(thetas[i + 1]), oy + np.sin(thetas[i + 1])
            verts = [(xa, ya, 0.0), (xb, yb, 0.0), (xb, yb, height), (xa, ya, height)]
            ax3d.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=[side_rgb],
                    edgecolors=[side_rgb],
                    linewidths=0.03,
                    alpha=1.0,
                )
            )
        for th in (thetas[0], thetas[-1]):
            x, y = ox + np.cos(th), oy + np.sin(th)
            verts = [(ox, oy, 0), (x, y, 0), (x, y, height), (ox, oy, height)]
            ax3d.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=[np.clip(side_rgb * 0.88, 0, 1)],
                    edgecolors="none",
                    alpha=1.0,
                )
            )
        theta0 -= dtheta

    ax_leg = fig.add_axes([bbox.x0, bbox.y0 + bbox.height - leg_h, bbox.width, leg_h], facecolor=BG)
    ax_leg.set_zorder(6)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    n = len(labels)
    xs = np.linspace(0.14, 0.86, n) if n > 1 else [0.5]
    for i, lab in enumerate(labels):
        lx = float(xs[i])
        ax_leg.plot(lx, 0.20, "o", color=colors[i], markersize=7.5, clip_on=False)
        ax_leg.text(lx, 0.72, lab, fontsize=7.4, color=C_INK, va="center", ha="center", fontweight="bold")
        ax_leg.text(
            lx,
            0.42,
            f"{int(sizes[i])} ({100.0 * fracs[i]:.0f}%)",
            fontsize=5.8,
            color=C_MUTED,
            va="center",
            ha="center",
        )

    ax_bot = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bot_h], facecolor=BG)
    ax_bot.set_zorder(6)
    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")
    ax_bot.text(0.5, 0.40, total_label, ha="center", va="center", fontsize=7.6, color=C_INK, fontweight="bold")


# ---------------------------------------------------------------------------
# Case cache
# ---------------------------------------------------------------------------


def _ensure_case_cache() -> dict:
    if CACHE.is_file():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    import torch
    from torch.utils.data import DataLoader

    from hnf.eeg_dataset import EEGDataset
    from tools.run_eeg_clinical_suite import _load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _REPO / "outputs/eeg/adftd_hnf_native_v3/best.pt"
    model, ckpt_args, _arch = _load_model(ckpt, device)
    model.eval()
    sr = int(ckpt_args.get("sample_rate", 128))
    epoch_sec = float(ckpt_args.get("epoch_sec", 10.0))
    ds = EEGDataset(
        data_dir=str(_REPO / "external_data/eeg_adftd"),
        split="test",
        seed=42,
        sample_rate=sr,
        epoch_sec=epoch_sec,
        stride_sec=epoch_sec,
        synthetic_if_missing=False,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    picked: dict[str, dict] = {}
    with torch.no_grad():
        for batch in loader:
            for i in range(batch["x"].size(0)):
                g = str(batch["clinical_group"][i])
                if g in picked:
                    continue
                x = batch["x"][i : i + 1].to(device)
                logits, aux = model(x, return_aux=True)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                xnp = x[0].cpu().numpy()
                wave = xnp.mean(0)
                rho = aux["rho"][0, :, 0].cpu().numpy()
                ch_pow = np.sqrt((xnp ** 2).mean(axis=1))
                # keep a few montage channels for stacked display
                # STANDARD_10_20: Fp1, Cz, T4, O1, O2
                ch_idx = [0, 9, 11, 17, 18]
                picked[g] = {
                    "sid": str(batch["subject_id"][i]),
                    "mmse": float(batch["mmse"][i]),
                    "wave": wave.astype(np.float32),
                    "channels": xnp[ch_idx].astype(np.float32),
                    "rho": rho.astype(np.float32),
                    "ch_pow": ch_pow.astype(np.float32),
                    "probs": probs.astype(np.float32),
                    "sr": np.float32(sr),
                }
            if len(picked) == 3:
                break
    if len(picked) < 3:
        raise SystemExit(f"Need HC/FTD/AD cases; got {list(picked)}")

    payload = {}
    for g, d in picked.items():
        for k, v in d.items():
            payload[f"{g}_{k}"] = v
    np.savez(CACHE, **payload)
    return payload


def _case_of(cache: dict, g: str) -> dict:
    out = {
        "sid": str(cache[f"{g}_sid"]),
        "mmse": float(cache[f"{g}_mmse"]),
        "wave": np.asarray(cache[f"{g}_wave"], dtype=np.float64),
        "rho": np.asarray(cache[f"{g}_rho"], dtype=np.float64),
        "ch_pow": np.asarray(cache[f"{g}_ch_pow"], dtype=np.float64),
        "probs": np.asarray(cache[f"{g}_probs"], dtype=np.float64),
        "sr": float(cache[f"{g}_sr"]),
    }
    key = f"{g}_channels"
    if key in cache:
        out["channels"] = np.asarray(cache[key], dtype=np.float64)
    return out


def _draw_scalp(ax, values: np.ndarray, cmap="YlOrRd") -> None:
    xyz = electrode_xyz()
    xy = xyz[:, :2]
    xi = np.linspace(-1.15, 1.15, 90)
    yi = np.linspace(-1.15, 1.15, 90)
    XX, YY = np.meshgrid(xi, yi)
    rbf = Rbf(xy[:, 0], xy[:, 1], values, function="multiquadric", smooth=0.15)
    ZZ = rbf(XX, YY)
    ZZ = np.ma.array(ZZ, mask=(XX**2 + YY**2 > 1.05**2))
    ax.contourf(XX, YY, ZZ, levels=14, cmap=cmap, zorder=1)
    ax.add_patch(Circle((0, 0), 1.05, fill=False, ec=C_INK, lw=0.75, zorder=3))
    ax.plot([0, -0.12, 0.12, 0], [1.05, 1.22, 1.22, 1.05], color=C_INK, lw=0.65, zorder=3)
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c="#111827", zorder=4, linewidths=0)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def panel_method(ax) -> None:
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.05)
    ax.axis("off")
    _panel_header(ax, "b", "From scalp EEG to a voltmeter-orthogonal medium leftover", y=1.02)

    # Top pipeline — equal height, clear gaps (no box overlap)
    top_y, top_h = 1.55, 1.25
    boxes = [
        (0.15, top_y, 2.00, top_h, "#F3F4F6", "19-ch EC EEG\n(10–20, 10 s)"),
        (2.55, top_y, 2.05, top_h, "#ECFDF5", "Voltmeter\nage · sex\nθ/α · bp α"),
        (5.00, top_y, 2.20, top_h, "#EFF6FF", "Frozen spatial\nprobe → $D_\\mathrm{eff}$\n+$\\rho(t)$"),
        (7.70, top_y, 4.00, top_h, "#F5F3FF", ""),
    ]
    for x, y, w, h, fc, txt in boxes:
        _box(ax, x, y, w, h, fc)
        if txt:
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2 * 1.3, color=C_INK, linespacing=1.15)

    t = np.linspace(0, 1, 220)
    wave = 0.55 * np.sin(18 * np.pi * t) * np.exp(-1.2 * (t - 0.45) ** 2) + 0.22 * np.sin(40 * np.pi * t)
    ax.plot(0.35 + t * 1.60, 1.78 + 0.22 * wave, color=C_INK, lw=0.65, zorder=4)

    ax.text(9.70, 2.50, "Train-only residualizer", ha="center", fontsize=7.6 * 1.3, fontweight="bold", color="#6D28D9")
    ax.text(
        9.70,
        1.95,
        r"leftover $= D_{\mathrm{eff}} - f_{\mathrm{train}}(\mathrm{volt})$"
        "\n"
        "apply OOS · no refit",
        ha="center",
        va="center",
        fontsize=7.0 * 1.3,
        color=C_INK,
    )

    mid = top_y + 0.5 * top_h
    for x0, x1 in [(2.15, 2.55), (4.60, 5.00), (7.20, 7.70)]:
        ax.annotate("", xy=(x1, mid), xytext=(x0, mid), arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.05))

    # Second-row controls: centered; font 1.3× original (7.0)
    gap = 0.40
    cw = (2.55, 2.55, 2.70)
    total = sum(cw) + 2 * gap
    x0 = 0.5 * (12.0 - total)
    ctrls = [
        (x0, 0.10, cw[0], 0.95, "#D1FAE5", "phase off\n(diffusion)"),
        (x0 + cw[0] + gap, 0.10, cw[1], 0.95, "#FEE2E2", "phase on\n(physics kill)"),
        (x0 + cw[0] + gap + cw[1] + gap, 0.10, cw[2], 0.95, "#DBEAFE", "two kernels\n→ one axis"),
    ]
    for x, y, w, h, fc, txt in ctrls:
        _box(ax, x, y, w, h, fc, r=0.06)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.0 * 1.3, color=C_INK, linespacing=1.15)


def panel_cases(fig, outer_spec, cache: dict) -> None:
    gs = GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=outer_spec,
        height_ratios=[0.07, 1.0],
        hspace=0.02,
        wspace=0.14,
    )
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis("off")
    ax_h.set_facecolor(BG)
    _panel_header(
        ax_h,
        "c",
        "Case epochs — stacked channels, ρ(t), scalp RMS (held-out subjects)",
        y=0.15,
    )

    ch_names = ["Fp1", "Cz", "T4", "O1", "O2"]
    order = ("HC", "FTD", "AD")
    for j, g in enumerate(order):
        case = _case_of(cache, g)
        col = GROUP_COLORS[g]
        # card background via a full-span axes
        ax_card = fig.add_subplot(gs[1, j])
        ax_card.set_xlim(0, 1)
        ax_card.set_ylim(0, 1)
        ax_card.axis("off")
        ax_card.add_patch(
            Rectangle((0.01, 0.01), 0.98, 0.98, fill=True, fc="#FCFCFD", ec="#E2E6EA", lw=0.85, zorder=0)
        )
        ax_card.text(
            0.04,
            0.955,
            f"{g}   {case['sid']}   MMSE {case['mmse']:.0f}",
            fontsize=8,
            fontweight="bold",
            color=col,
            va="top",
            ha="left",
            zorder=2,
            transform=ax_card.transAxes,
        )

        pos = ax_card.get_position()
        # nested regions inside card (figure coords)
        x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height
        y_body = y0 + 0.06 * h
        h_body = 0.86 * h
        # stacked channels / rho / topo+band — more room for scalp
        ax_ch = fig.add_axes([x0 + 0.12 * w, y_body + 0.48 * h_body, 0.82 * w, 0.48 * h_body])
        ax_r = fig.add_axes([x0 + 0.12 * w, y_body + 0.30 * h_body, 0.82 * w, 0.14 * h_body])
        ax_t = fig.add_axes([x0 + 0.06 * w, y_body + 0.02 * h_body, 0.44 * w, 0.26 * h_body])
        ax_b = fig.add_axes([x0 + 0.54 * w, y_body + 0.04 * h_body, 0.40 * w, 0.22 * h_body])

        sr = case["sr"]
        chs = case.get("channels")
        if chs is None:
            chs = case["wave"][None, :]
            names = ["mean"]
        else:
            names = ch_names[: chs.shape[0]]
        t = np.arange(chs.shape[1]) / sr
        _clean(ax_ch)
        for i, name in enumerate(names):
            sig = chs[i]
            sig = sig / (np.max(np.abs(sig)) + 1e-8)
            yoff = (len(names) - 1 - i) * 1.35
            ax_ch.plot(t, sig * 0.55 + yoff, color=C_INK, lw=0.45)
            ax_ch.text(
                -0.02,
                yoff,
                name,
                fontsize=5.6,
                color=C_MUTED,
                ha="right",
                va="center",
                clip_on=False,
                transform=ax_ch.get_yaxis_transform(),
            )
        ax_ch.set_xlim(0, t[-1])
        ax_ch.set_ylim(-0.9, len(names) * 1.35 - 0.35)
        ax_ch.set_yticks([])
        ax_ch.tick_params(labelsize=5.5)
        ax_ch.set_title("10 s epoch", fontsize=6.5, color=C_MUTED, loc="left", pad=1)

        rn = case["rho"] / (np.max(case["rho"]) + 1e-8)
        tr = np.linspace(0, t[-1], len(rn))
        _clean(ax_r)
        ax_r.fill_between(tr, 0, rn, color=col, alpha=0.28, lw=0)
        ax_r.plot(tr, rn, color=col, lw=0.95)
        ax_r.set_xlim(0, t[-1])
        ax_r.set_ylim(0, 1.15)
        ax_r.set_yticks([])
        ax_r.set_ylabel(r"$\rho$", fontsize=6.5)
        ax_r.tick_params(labelsize=5.5)

        _draw_scalp(ax_t, case["ch_pow"], cmap="YlOrRd")

        bands = ["δ", "θ", "α", "β"]
        spec = np.abs(np.fft.rfft(case["wave"])) ** 2
        freqs = np.fft.rfftfreq(len(case["wave"]), d=1.0 / sr)
        edges = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
        bp = []
        for lo, hi in edges:
            msk = (freqs >= lo) & (freqs < hi)
            bp.append(float(spec[msk].mean()) if msk.any() else 0.0)
        bp = np.asarray(bp)
        bp = bp / (bp.sum() + 1e-8)
        _clean(ax_b)
        ax_b.bar(bands, bp, color=[C_AD, C_TEAL, C_GOLD, C_FTD], width=0.65)
        ax_b.set_ylim(0, max(0.55, float(bp.max()) * 1.3))
        ax_b.tick_params(labelsize=5.5)
        ax_b.set_title("band", fontsize=6, color=C_MUTED, pad=0)


def panel_cognition(ax, v3: pd.DataFrame) -> None:
    _clean(ax)
    _panel_header(ax, "d", "Held-out cognition vs leftover", y=1.06)
    tr = v3[v3.split.eq("train")].copy()
    vt = v3[v3.split.isin(["val", "test"])].copy()
    # Train-fit residualizer for MMSE (same protocol as D_eff leftover).
    X_tr = np.column_stack(
        [
            tr["age"].to_numpy(float),
            sex_to_float(tr["gender"].tolist()),
            tr["theta_alpha_ratio"].to_numpy(float),
            tr["bp_alpha"].to_numpy(float),
        ]
    )
    _, _, beta_m = residualize(tr["mmse"].to_numpy(float), X_tr)
    X_vt = np.column_stack(
        [
            vt["age"].to_numpy(float),
            sex_to_float(vt["gender"].tolist()),
            vt["theta_alpha_ratio"].to_numpy(float),
            vt["bp_alpha"].to_numpy(float),
        ]
    )
    mmse_r = vt["mmse"].to_numpy(float) - np.column_stack([np.ones(len(vt)), X_vt]) @ beta_m
    lef = vt["D_eff_res_trainfit"].to_numpy(float)
    for g in ("HC", "FTD", "AD"):
        idx = vt.clinical_group.to_numpy() == g
        ax.scatter(lef[idx], mmse_r[idx], s=30, c=GROUP_COLORS[g], label=g, alpha=0.9, edgecolors="none", zorder=3)
    m = np.isfinite(lef) & np.isfinite(mmse_r)
    coef = np.polyfit(lef[m], mmse_r[m], 1)
    xs = np.linspace(np.nanpercentile(lef[m], 5), np.nanpercentile(lef[m], 95), 40)
    ax.plot(xs, coef[0] * xs + coef[1], color=C_TEAL, lw=1.25, zorder=2)
    ax.axhline(0, color=C_LINE, lw=0.6)
    ax.axvline(0, color=C_LINE, lw=0.6)
    r_s, p_s = spearman_r(lef, mmse_r)
    ax.set_xlabel(r"leftover $D_{\mathrm{eff}}$ (train-fit)", fontsize=7.5)
    ax.set_ylabel("MMSE | voltmeter (train-fit)", fontsize=7.5)
    # stats in lower-left clear zone
    ax.text(
        0.03,
        0.05,
        rf"$\rho$={r_s:.2f}   $p$={p_s:.2g}   $n$={int(m.sum())}",
        transform=ax.transAxes,
        fontsize=7,
        color=C_MUTED,
        va="bottom",
        zorder=5,
    )
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.25, borderaxespad=0.2)


def panel_physics(ax, v3, an, on) -> None:
    _clean(ax)
    _panel_header(ax, "e", "Probe-physics control", y=1.06)

    def delta_all(df: pd.DataFrame) -> float:
        y = df.mmse.to_numpy(float)
        ok = np.isfinite(y)
        X = np.column_stack(
            [
                df.age.to_numpy(float),
                sex_to_float(df.gender),
                df.theta_alpha_ratio.to_numpy(float),
                df.bp_alpha.to_numpy(float),
            ]
        )
        pr = df.D_eff_res_trainfit.to_numpy(float)
        _, r0, _ = residualize(y[ok], X[ok])
        _, r1, _ = residualize(y[ok], np.c_[X[ok], pr[ok]])
        return float(r1 - r0)

    names = ["Fresnel", "diff.\nphase off", "diff.\nphase on"]
    vals = [delta_all(v3), delta_all(an), delta_all(on)]
    cols = [C_TEAL, C_RING, C_MUTED]
    bars = ax.bar(np.arange(3), vals, color=cols, width=0.62, zorder=3)
    ax.axhline(0, color=C_LINE, lw=0.7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(r"MMSE $\Delta R^{2}$ | voltmeter", fontsize=7.5)
    ymax = max(vals) * 1.45
    ax.set_ylim(min(-0.015, min(vals) * 1.3), ymax)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.012,
            f"{v:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=C_INK,
            clip_on=False,
        )


def panel_dual(ax, v3, an) -> None:
    _clean(ax)
    _panel_header(ax, "f", "Two kernels, one leftover axis", y=1.06)
    m = v3.merge(
        an[["subject_id", "D_eff_res_trainfit"]],
        on="subject_id",
        suffixes=("_A", "_B"),
    )
    vt = m[m.split.isin(["val", "test"])]
    for g in ("HC", "FTD", "AD"):
        d = vt[vt.clinical_group == g]
        ax.scatter(
            d.D_eff_res_trainfit_A,
            d.D_eff_res_trainfit_B,
            s=26,
            c=GROUP_COLORS[g],
            alpha=0.88,
            edgecolors="none",
            label=g,
            zorder=3,
        )
    x = vt.D_eff_res_trainfit_A.to_numpy(float)
    y = vt.D_eff_res_trainfit_B.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    r, p = spearman_r(x, y)
    lim = np.nanpercentile(np.r_[x[ok], y[ok]], [2, 98])
    ax.plot(lim, lim, color=C_MUTED, lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("Fresnel leftover", fontsize=7.5)
    ax.set_ylabel("Anisotropic leftover", fontsize=7.5)
    ax.text(
        0.03,
        0.05,
        rf"held-out $\rho$={r:.2f}   $p$={p:.1g}   $n$={int(ok.sum())}",
        transform=ax.transAxes,
        fontsize=7,
        color=C_MUTED,
        va="bottom",
    )


def lemon_ols_leftover(lr: pd.DataFrame) -> np.ndarray:
    y = lr["native_v3_ec_D_eff"].to_numpy(float)
    X = np.column_stack(
        [
            lr["age"].to_numpy(float),
            sex_to_float(lr["sex"].tolist()),
            lr["native_v3_ec_theta_alpha_ratio"].to_numpy(float),
            lr["native_v3_ec_bp_alpha"].to_numpy(float),
        ]
    )
    resid, _, _ = residualize(y, X)
    return resid


def panel_discovery(ax, v3, lr) -> None:
    _clean(ax)
    _panel_header(ax, "g", "Discovery — pathology-coupled", y=1.06, title_dx_pt=15.0)

    lef_a = v3["D_eff_res_trainfit"].to_numpy(float)
    mmse = v3["mmse"].to_numpy(float)
    demo = np.column_stack([v3["age"].to_numpy(float), sex_to_float(v3["gender"].tolist())])
    mmse_d, _, _ = residualize(mmse, demo)
    lef_d, _, _ = residualize(lef_a, demo)
    r_dis, _ = spearman_r(lef_d, mmse_d)

    lef_l = lemon_ols_leftover(lr)
    age = lr["age"].to_numpy(float)
    gm = lr["gm_icv"].to_numpy(float)
    tmt = lr["tmt_b"].to_numpy(float)
    sex = sex_to_float(lr["sex"].tolist())
    demo_l = np.column_stack([age, sex])
    r_age, _ = spearman_r(lef_l, age)
    r_gm, _ = spearman_r(residualize(lef_l, demo_l)[0], residualize(gm, demo_l)[0])
    r_tmt, _ = spearman_r(residualize(lef_l, demo_l)[0], residualize(tmt, demo_l)[0])

    labels = ["AHEPA\nMMSE|demo", "LEMON\nage", "LEMON\nGM|demo", "LEMON\nTMT|demo"]
    vals = [abs(r_dis), abs(r_age), abs(r_gm), abs(r_tmt)]
    cols = [C_TEAL, C_MUTED, C_MUTED, C_MUTED]
    bars = ax.bar(np.arange(4), vals, color=cols, width=0.62, zorder=3)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(r"|Spearman $\rho$|", fontsize=7.5)
    ax.set_ylim(0, max(0.48, max(vals) * 1.55))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # tags above plot area, not on bars
    ax.text(0.0, 1.02, "disease", transform=ax.transAxes, fontsize=6.5, color=C_TEAL, fontweight="bold", ha="left", va="bottom", clip_on=False)
    ax.text(1.0, 1.02, "healthy null", transform=ax.transAxes, fontsize=6.5, color=C_MUTED, fontweight="bold", ha="right", va="bottom", clip_on=False)


def main() -> None:
    _style()
    v3 = pd.read_csv(_REPO / "outputs/eeg/probe_publishable/subjects_native_v3_trainfit_residual.csv")
    an = pd.read_csv(_REPO / "outputs/eeg/probe_publishable/subjects_aniso_phase_off_trainfit_residual.csv")
    on = pd.read_csv(_REPO / "outputs/eeg/probe_publishable/subjects_aniso_phase_on_trainfit_residual.csv")
    lr = pd.read_csv(_REPO / "outputs/eeg/lemon_probe/subjects_probe.csv")

    # refresh case cache if missing channel stacks
    if CACHE.is_file():
        z = np.load(CACHE, allow_pickle=True)
        if "HC_channels" not in z.files:
            CACHE.unlink()
    cache = _ensure_case_cache()

    n_hc = int((v3.clinical_group == "HC").sum())
    n_ftd = int((v3.clinical_group == "FTD").sum())
    n_ad = int((v3.clinical_group == "AD").sum())
    n_young = int((lr.age < 60).sum())
    n_old = int((lr.age >= 60).sum())

    fig = plt.figure(figsize=(13.0, 10.8), facecolor=BG)
    # Explicit spacers: small gap after row1, larger gap between row2 and row3
    gs = GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[1.08, 0.05, 1.52, 0.26, 1.12],
        hspace=0.0,
        left=0.050,
        right=0.935,
        top=0.962,
        bottom=0.050,
    )

    # Row 1: a pies | b method
    gs0 = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs[0],
        height_ratios=[0.14, 1.0],
        width_ratios=[1.25, 1.75],
        hspace=0.04,
        wspace=0.12,
    )
    ax_ha = fig.add_subplot(gs0[0, 0])
    ax_hb = fig.add_subplot(gs0[0, 1])
    for axh in (ax_ha, ax_hb):
        axh.set_facecolor(BG)
        axh.axis("off")
    _panel_header(ax_ha, "a", "Cohorts — clinical vs healthy lifespan", y=0.15)
    ax_hb.text(0.0, 0.15, "", transform=ax_hb.transAxes)

    gs_pie = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1, 0], wspace=0.14)
    ax_pie1 = fig.add_subplot(gs_pie[0, 0])
    ax_pie2 = fig.add_subplot(gs_pie[0, 1])
    ax_b = fig.add_subplot(gs0[1, 1])

    _boom_pie_3d(
        fig,
        ax_pie1,
        ["HC", "FTD", "AD"],
        [n_hc, n_ftd, n_ad],
        [C_HC, C_FTD, C_AD],
        f"AHEPA  n={n_hc + n_ftd + n_ad}",
    )
    _boom_pie_3d(
        fig,
        ax_pie2,
        ["young", "old"],
        [n_young, n_old],
        [C_YOUNG, C_OLD],
        f"LEMON  n={n_young + n_old}",
    )
    panel_method(ax_b)

    # Row 2: cases  (gs[1] is thin spacer)
    panel_cases(fig, gs[2], cache)

    # Row 3: outcomes + discovery  (gs[3] is larger spacer)  widths 3:2:3:2
    gs2 = GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=gs[4],
        wspace=0.28,
        width_ratios=[3, 2, 3, 2],
    )
    ax_d = fig.add_subplot(gs2[0, 0])
    ax_e = fig.add_subplot(gs2[0, 1])
    ax_f = fig.add_subplot(gs2[0, 2])
    ax_g = fig.add_subplot(gs2[0, 3])
    panel_cognition(ax_d, v3)
    panel_physics(ax_e, v3, an, on)
    panel_dual(ax_f, v3, an)
    panel_discovery(ax_g, v3, lr)

    fig.suptitle(
        "Voltmeter-orthogonal EEG medium leftover is pathology-coupled",
        fontsize=11.5,
        fontweight="bold",
        color=C_INK,
        x=0.055,
        ha="left",
        y=0.985,
    )

    fig.canvas.draw()
    ax_ha.set_zorder(90)
    ax_ha.patch.set_visible(True)
    ax_ha.patch.set_facecolor(BG)
    ax_ha.patch.set_alpha(1.0)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "eeg_impact_discovery"
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(f"[fig] → {OUT / f'{stem}.png'}")


if __name__ == "__main__":
    main()
