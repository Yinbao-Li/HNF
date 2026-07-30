#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figure: interpretable latent variables in the seismic domain.

Panels
------
(a) representative 3-component waveform with P/S arrivals (dashed)
(b) learned density field ρ(t) — rises at P, peaks in S window
(c) kernel row at ground-truth P index — causal support before onset
(d) P and S pick probability curves

Runs on **CPU** by default so it does not contend with an ongoing GPU train.

Example
-------
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python tools/plot_seismic_latent_figure.py \\
    --device cpu --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seismic interpretable latent figure")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt",
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scan", type=int, default=80, help="candidates to score for a clean example")
    p.add_argument("--trace-name", default="", help="optional fixed STEAD trace_name")
    p.add_argument(
        "--output",
        default="docs/figures/seismic_interpretable_latents.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    return p.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_label(ax, letter: str, x: float = -0.07, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def _mark_ps(ax, t_sec: np.ndarray, p_idx: int, s_idx: int) -> None:
    if p_idx >= 0:
        ax.axvline(t_sec[p_idx], color="#4C7A5A", ls="--", lw=0.9, alpha=0.85, zorder=3)
    if s_idx >= 0:
        ax.axvline(t_sec[s_idx], color="#8B5E3C", ls="--", lw=0.9, alpha=0.85, zorder=3)


def _score_example(
    *,
    x: np.ndarray,
    t_sec: np.ndarray,
    rho: np.ndarray,
    p_prob: np.ndarray,
    s_prob: np.ndarray,
    kernel: np.ndarray,
    p_true: int,
    s_true: int,
    thr: float,
) -> float:
    """Prefer clear P/S, ρ rise at P, and causal kernel mass before onset."""
    if p_true < 8 or s_true <= p_true + 5:
        return -1e9
    gap = s_true - p_true
    if gap < 15 or gap > 350:
        return -1e9
    p_peak = float(p_prob[p_true])
    s_peak = float(s_prob[s_true])
    if p_peak < thr or s_peak < thr * 0.8:
        return -1e9
    # ρ: mean in S window vs pre-P noise
    pre = rho[max(0, p_true - 40) : p_true]
    sw = rho[p_true : s_true + max(10, gap // 4)]
    if pre.size < 5 or sw.size < 5:
        return -1e9
    rho_ratio = float(sw.mean() / (pre.mean() + 1e-6))
    # kernel causal mass: fraction of |K| strictly before P
    k = np.abs(kernel)
    k_sum = float(k.sum()) + 1e-12
    causal_frac = float(k[:p_true].sum() / k_sum)
    near = float(k[max(0, p_true - 40) : p_true].sum() / k_sum)
    # waveform SNR proxy
    amp = np.sqrt((x**2).mean(axis=1))
    snr = float(amp[p_true:s_true].mean() / (amp[: max(1, p_true)].mean() + 1e-6))
    return (
        2.0 * p_peak
        + 1.5 * s_peak
        + 1.2 * np.clip(rho_ratio, 0, 5)
        + 2.5 * causal_frac
        + 1.5 * near
        + 0.4 * np.clip(snr, 0, 8)
    )


def _select_example(model, ds, device, args) -> dict:
    rng = np.random.default_rng(args.seed)
    # Prefer events with valid P/S labels
    event_ids = [i for i, r in enumerate(ds.refs) if r.is_event and r.p_sample is not None and r.s_sample is not None]
    if args.trace_name:
        for i, r in enumerate(ds.refs):
            if str(r.trace_name) == str(args.trace_name):
                event_ids = [i]
                break
        else:
            raise SystemExit(f"trace_name not found: {args.trace_name}")

    cand = event_ids
    if len(cand) > args.scan and not args.trace_name:
        cand = list(rng.choice(cand, size=args.scan, replace=False))

    best = None
    best_score = -1e18
    for i in cand:
        sample = ds[int(i)]
        if float(sample["p_valid"]) <= 0 or float(sample["s_valid"]) <= 0:
            continue
        x = sample["x"].unsqueeze(0).to(device)
        t = sample["t"].unsqueeze(0).to(device)
        p_true = int(sample["p_idx"].item())
        s_true = int(sample["s_idx"].item())
        with torch.no_grad():
            out = model.forward_explain(
                x,
                t,
                include_kernel_row=True,
                kernel_row_idx=p_true,
                kernel_branch="p",
            )
        x_np = sample["x"].cpu().numpy()
        t_sec = sample["t"][:, 0].cpu().numpy()
        rho = out["rho"][0].detach().cpu().numpy()
        p_prob = torch.sigmoid(out["p"][0]).cpu().numpy()
        s_prob = torch.sigmoid(out["s"][0]).cpu().numpy()
        kernel = out["kernel_contrib"][0].detach().cpu().numpy()
        score = _score_example(
            x=x_np,
            t_sec=t_sec,
            rho=rho,
            p_prob=p_prob,
            s_prob=s_prob,
            kernel=kernel,
            p_true=p_true,
            s_true=s_true,
            thr=args.pick_threshold,
        )
        if score > best_score:
            best_score = score
            best = {
                "idx": int(i),
                "trace_name": str(sample["trace_name"]),
                "x": x_np,
                "t_sec": t_sec,
                "rho": rho,
                "p_prob": p_prob,
                "s_prob": s_prob,
                "kernel": kernel,
                "wave_energy": out["wave_energy"][0].detach().cpu().numpy(),
                "p_true": p_true,
                "s_true": s_true,
                "p_pred": int(p_prob.argmax()),
                "s_pred": int(s_prob.argmax()),
                "score": float(score),
            }
    if best is None:
        raise SystemExit("no suitable example found")
    print(
        f"[latent-fig] selected idx={best['idx']}  name={best['trace_name']}  "
        f"score={best['score']:.3f}  P={best['t_sec'][best['p_true']]:.2f}s  "
        f"S={best['t_sec'][best['s_true']]:.2f}s",
        flush=True,
    )
    return best


def plot_figure(ex: dict, out_path: Path, dpi: int) -> None:
    t = ex["t_sec"]
    p_true, s_true = ex["p_true"], ex["s_true"]
    x = ex["x"]
    # demean / scale per channel for display
    xd = x - x.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(xd)) + 1e-8
    xd = xd / scale

    fig = plt.figure(figsize=(7.2, 7.6), dpi=dpi)
    gs = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1.15, 0.85, 0.95, 0.95],
        hspace=0.18,
        left=0.11,
        right=0.98,
        top=0.96,
        bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_d = fig.add_subplot(gs[3, 0], sharex=ax_a)

    # ---- (a) waveform ----
    _panel_label(ax_a, "a")
    colors = ["#1B4F72", "#2E86AB", "#6C757D"]
    names = ["E", "N", "Z"]
    offsets = [1.15, 0.0, -1.15]
    for i in range(3):
        ax_a.plot(t, xd[:, i] + offsets[i], color=colors[i], lw=0.7, solid_capstyle="round", label=names[i])
    _mark_ps(ax_a, t, p_true, s_true)
    ax_a.text(t[p_true] + 0.4, 2.05, "P", color="#4C7A5A", fontsize=8, fontweight="bold")
    ax_a.text(t[s_true] + 0.4, 2.05, "S", color="#8B5E3C", fontsize=8, fontweight="bold")
    ax_a.set_ylabel("Normalised\namplitude")
    ax_a.set_title("Representative three-component waveform with P and S arrivals", loc="left", pad=3)
    ax_a.set_yticks(offsets)
    ax_a.set_yticklabels(names)
    ax_a.set_ylim(-2.35, 2.45)
    ax_a.legend(loc="upper right", ncol=3, handlelength=1.2)

    # ---- (b) rho ----
    _panel_label(ax_b, "b")
    rho = ex["rho"]
    energy = ex["wave_energy"]
    energy_n = energy / (energy.max() + 1e-8)
    rho_n = rho / (rho.max() + 1e-8)
    ax_b.fill_between(t, 0, energy_n, color="#C8C3B8", alpha=0.45, linewidth=0, label="waveform energy")
    ax_b.plot(t, rho_n, color="#8B4513", lw=1.25, label="ρ(t)")
    _mark_ps(ax_b, t, p_true, s_true)
    # annotate rise / peak
    ax_b.annotate(
        "rises at P",
        xy=(t[p_true], rho_n[p_true]),
        xytext=(t[p_true] + 3.5, min(0.95, rho_n[p_true] + 0.28)),
        fontsize=7,
        color="#8B4513",
        arrowprops=dict(arrowstyle="-", color="#8B4513", lw=0.7),
    )
    s_win = slice(p_true, min(len(t), s_true + max(5, (s_true - p_true) // 3)))
    peak_i = int(p_true + np.argmax(rho_n[s_win]))
    ax_b.annotate(
        "peaks in S window",
        xy=(t[peak_i], rho_n[peak_i]),
        xytext=(t[peak_i] + 2.5, max(0.55, rho_n[peak_i] - 0.25)),
        fontsize=7,
        color="#8B4513",
        arrowprops=dict(arrowstyle="-", color="#8B4513", lw=0.7),
    )
    ax_b.set_ylabel("Normalised\nρ / energy")
    ax_b.set_title("Learned density field ρ(t) tracks waveform energy", loc="left", pad=3)
    ax_b.set_ylim(-0.02, 1.18)
    ax_b.legend(loc="upper right", ncol=2)

    # ---- (c) kernel row ----
    _panel_label(ax_c, "c")
    k = np.abs(ex["kernel"])
    k = k / (k.max() + 1e-12)
    ax_c.fill_between(t, 0, k, color="#5E3C99", alpha=0.18, linewidth=0)
    ax_c.plot(t, k, color="#5E3C99", lw=1.15, label="|K[P, :]|")
    _mark_ps(ax_c, t, p_true, s_true)
    # shade causal past
    ax_c.axvspan(t[0], t[p_true], color="#5E3C99", alpha=0.06, zorder=0)
    causal_frac = float(k[:p_true].sum() / (k.sum() + 1e-12))
    ax_c.annotate(
        f"causal support before onset\n({100 * causal_frac:.0f}% of |K| mass)",
        xy=(t[max(0, p_true - 15)], float(k[max(0, p_true - 15) : p_true].max() if p_true > 0 else 0)),
        xytext=(max(t[0] + 1.0, t[p_true] - 18), 0.78),
        fontsize=7,
        color="#5E3C99",
        arrowprops=dict(arrowstyle="-", color="#5E3C99", lw=0.7),
    )
    ax_c.set_ylabel("Normalised\n|kernel|")
    ax_c.set_title("Kernel row at ground-truth P index", loc="left", pad=3)
    ax_c.set_ylim(-0.02, 1.18)
    ax_c.legend(loc="upper right")

    # ---- (d) pick probs ----
    _panel_label(ax_d, "d")
    ax_d.plot(t, ex["p_prob"], color="#0072B2", lw=1.25, label="P pick probability")
    ax_d.plot(t, ex["s_prob"], color="#D55E00", lw=1.25, label="S pick probability")
    _mark_ps(ax_d, t, p_true, s_true)
    ax_d.scatter([t[ex["p_pred"]]], [ex["p_prob"][ex["p_pred"]]], s=22, color="#0072B2", zorder=4)
    ax_d.scatter([t[ex["s_pred"]]], [ex["s_prob"][ex["s_pred"]]], s=22, color="#D55E00", zorder=4)
    ax_d.set_ylabel("Probability")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_title("P and S pick probability curves", loc="left", pad=3)
    ax_d.set_ylim(-0.02, 1.12)
    ax_d.legend(loc="upper right", ncol=2)

    # shared x limits: focus around P–S with margins
    t0 = max(0.0, t[p_true] - 8.0)
    t1 = min(float(t[-1]), t[s_true] + 12.0)
    ax_a.set_xlim(t0, t1)
    for ax in (ax_a, ax_b, ax_c):
        plt.setp(ax.get_xticklabels(), visible=False)

    # shared legend note for dashed arrivals
    handles = [
        Line2D([0], [0], color="#4C7A5A", ls="--", lw=0.9, label="P arrival (GT)"),
        Line2D([0], [0], color="#8B5E3C", ls="--", lw=0.9, label="S arrival (GT)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.55, 0.995),
        frameon=False,
        fontsize=7.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[latent-fig] wrote {out_path}", flush=True)
    print(f"[latent-fig] wrote {out_path.with_suffix('.pdf')}", flush=True)


def main() -> None:
    args = parse_args()
    _style()
    device = torch.device("cpu" if str(args.device).lower() == "cpu" else args.device)
    print(f"[latent-fig] device={device}", flush=True)

    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.seq_len,
        max_event_traces=None,
        max_noise_traces=0,
        seed=args.seed,
    )
    ex = _select_example(model, ds, device, args)
    meta = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "trace_name": ex["trace_name"],
        "dataset_idx": ex["idx"],
        "p_true_sec": float(ex["t_sec"][ex["p_true"]]),
        "s_true_sec": float(ex["t_sec"][ex["s_true"]]),
        "p_pred_sec": float(ex["t_sec"][ex["p_pred"]]),
        "s_pred_sec": float(ex["t_sec"][ex["s_pred"]]),
        "score": ex["score"],
    }
    out = Path(args.output)
    plot_figure(ex, out, args.dpi)
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[latent-fig] meta → {meta_path}", flush=True)


if __name__ == "__main__":
    main()
