#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical-neuron probing suite for HNF picking (README Part II.2).

(1) Causal-chain tracking — freeze weights; record rho(t) and per-layer
    wavefield energy from shallow → deep; quantify peak sharpening onto P/S.
(2) Counterfactual ρ scrubbing — zero / damp mid-pipeline rho near S (or P)
    onset; measure ΔP / ΔS peak probabilities (causal role of rho).
(3) Anomaly attribution — false-P noise cases; inspect |K| row peak location
    vs physically plausible onset windows.

Usage:
  python scripts/interpret/run_probing_suite.py --device cuda \\
    --checkpoint outputs/run28/28_ms_fresnel_phys_20ep/best.pt \\
    --copy-to-docs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.picking_metrics import idx_to_sec
from hnf.stead_picking_dataset import STEADPickingDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF physical-neuron probing suite")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt",
    )
    p.add_argument("--output-dir", default="outputs/probing_suite_run28")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--n-chain", type=int, default=8)
    p.add_argument("--n-counterfactual", type=int, default=24)
    p.add_argument("--n-anomaly", type=int, default=12)
    p.add_argument("--scrub-radius-sec", type=float, default=0.6)
    p.add_argument("--copy-to-docs", action="store_true")
    return p.parse_args()


def _peak_width(curve: np.ndarray, center: int, frac: float = 0.5) -> float:
    """Half-max width (samples) around a preferred center index."""
    if curve.size == 0:
        return float("nan")
    c = int(np.clip(center, 0, len(curve) - 1))
    # Prefer local max near center, else global
    lo, hi = max(0, c - 40), min(len(curve), c + 41)
    local = lo + int(np.argmax(curve[lo:hi]))
    peak = float(curve[local])
    if peak <= 1e-8:
        return float("nan")
    thr = peak * frac
    left = local
    while left > 0 and curve[left] >= thr:
        left -= 1
    right = local
    while right < len(curve) - 1 and curve[right] >= thr:
        right += 1
    return float(right - left)


def _energy_1d(h_real: torch.Tensor, h_imag: torch.Tensor) -> torch.Tensor:
    """(B, T) wavefield energy."""
    return (h_real.pow(2) + h_imag.pow(2)).mean(dim=-1)


@torch.no_grad()
def forward_with_layer_trace(model, x: torch.Tensor, t: torch.Tensor) -> dict:
    """Like forward_pick_only, but collect energy after each Huygens stage."""
    bypass = getattr(model, "bypass_noise_cancel", True)
    model.bypass_noise_cancel = bypass
    _x_det, x_pick, nc_out = model._apply_noise_cancel(x, t)
    rho = model.medium_net(x_pick)
    h_real = model.source_embed(x_pick)
    h_imag = torch.zeros_like(h_real)

    stages: list[tuple[str, np.ndarray]] = [
        ("embed", _energy_1d(h_real, h_imag)[0].detach().cpu().numpy()),
    ]

    # Shared encoder (may be multi-scale): record after full shared block
    h_real, h_imag = model._encode_shared_wavefield(h_real, h_imag, t=t, rho=rho)
    stages.append(("shared", _energy_1d(h_real, h_imag)[0].detach().cpu().numpy()))

    if nc_out is not None and model.noise_cue_adapter is not None:
        cue, gate = model.noise_cue_adapter(
            x, nc_out["n_sim"], nc_out["u_denoised"], nc_out["s_noise"]
        )
        h_real = h_real + gate * cue
        stages.append(("noise_cue", _energy_1d(h_real, h_imag)[0].detach().cpu().numpy()))

    h_p, h_i = h_real, h_imag
    for i, layer in enumerate(model.p_layers):
        h_p, h_i = layer(h_p, h_i, t=t, rho=rho)
        stages.append((f"p_layer_{i}", _energy_1d(h_p, h_i)[0].detach().cpu().numpy()))
    p = model.p_pick_head(h_p, h_i)

    h_s, h_si = h_real, h_imag
    for i, layer in enumerate(model.s_layers):
        h_s, h_si = layer(h_s, h_si, t=t, rho=rho)
        stages.append((f"s_layer_{i}", _energy_1d(h_s, h_si)[0].detach().cpu().numpy()))
    s = model.s_pick_head(h_s, h_si)

    return {
        "rho": rho.squeeze(-1)[0].detach().cpu().numpy(),
        "p": p[0].detach().cpu(),
        "s": s[0].detach().cpu(),
        "stages": stages,
        "t_sec": t[0, :, 0].detach().cpu().numpy(),
    }


@torch.no_grad()
def forward_with_rho_override(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    rho_override: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Forward pick path with optional rho(t) override (B,T,1)."""
    _x_det, x_pick, nc_out = model._apply_noise_cancel(x, t)
    rho = model.medium_net(x_pick) if rho_override is None else rho_override
    h_real = model.source_embed(x_pick)
    h_imag = torch.zeros_like(h_real)
    h_real, h_imag = model._encode_shared_wavefield(h_real, h_imag, t=t, rho=rho)
    if nc_out is not None and model.noise_cue_adapter is not None:
        cue, gate = model.noise_cue_adapter(
            x, nc_out["n_sim"], nc_out["u_denoised"], nc_out["s_noise"]
        )
        h_real = h_real + gate * cue
    p_real, p_imag = model._propagate(h_real, h_imag, model.p_layers, t, rho)
    s_real, s_imag = model._propagate(h_real, h_imag, model.s_layers, t, rho)
    p = model.p_pick_head(p_real, p_imag)
    s = model.s_pick_head(s_real, s_imag)
    return {"p": p, "s": s, "rho": rho.squeeze(-1)}


def run_causal_chain(model, device, out_dir: Path, seq_len: int, n_cases: int) -> dict:
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=400)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    widths_embed: list[float] = []
    widths_deep: list[float] = []
    plotted = 0
    chain_dir = out_dir / "causal_chain"
    chain_dir.mkdir(exist_ok=True)

    for batch in loader:
        if plotted >= n_cases:
            break
        if float(batch["det"][0]) <= 0.5 or int(batch["p_valid"][0]) <= 0:
            continue
        x, t = batch["x"].to(device), batch["t"].to(device)
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0]) if int(batch["s_valid"][0]) > 0 else p_idx
        trace = forward_with_layer_trace(model, x, t)
        t_sec = trace["t_sec"]
        rho = trace["rho"]
        stages = trace["stages"]

        # Width of early energy vs late P-branch energy around GT P
        early = stages[0][1]
        deep_p = None
        for name, eng in stages:
            if name.startswith("p_layer_"):
                deep_p = eng
        if deep_p is None:
            deep_p = stages[-1][1]
        w0 = _peak_width(early, p_idx)
        w1 = _peak_width(deep_p, p_idx)
        widths_embed.append(w0)
        widths_deep.append(w1)

        fig, axes = plt.subplots(len(stages) + 2, 1, figsize=(10, 1.4 * (len(stages) + 2)), sharex=True)
        z = x[0, :, 2].detach().cpu().numpy()
        axes[0].plot(t_sec, z, color="0.25", lw=0.8)
        axes[0].axvline(idx_to_sec(p_idx, seq_len), color="C2", ls="--", alpha=0.7)
        axes[0].axvline(idx_to_sec(s_idx, seq_len), color="C1", ls="--", alpha=0.7)
        axes[0].set_ylabel("Z")
        name = batch["trace_name"][0] if isinstance(batch["trace_name"], (list, tuple)) else str(batch["trace_name"])
        axes[0].set_title(f"Causal chain — {name}")

        axes[1].plot(t_sec, rho, color="C3")
        axes[1].set_ylabel("rho")

        for ax, (lab, eng) in zip(axes[2:], stages):
            ax.plot(t_sec, eng, color="C0", lw=0.9)
            ax.axvline(idx_to_sec(p_idx, seq_len), color="C2", ls="--", alpha=0.5)
            ax.axvline(idx_to_sec(s_idx, seq_len), color="C1", ls="--", alpha=0.5)
            ax.set_ylabel(lab, fontsize=8)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(chain_dir / f"chain_{plotted:02d}.png", dpi=140)
        plt.close(fig)
        plotted += 1

    # Summary: shallow vs deep width
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(widths_embed, widths_deep, alpha=0.8)
    lim = max(max(widths_embed + [1]), max(widths_deep + [1])) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4)
    ax.set_xlabel("embed energy peak width (samples)")
    ax.set_ylabel("deep P-layer energy peak width (samples)")
    ax.set_title("Focusing: shallow → deep (below diagonal = sharper)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    summary = out_dir / "causal_chain_focusing_scatter.png"
    fig.savefig(summary, dpi=150)
    plt.close(fig)

    sharper = sum(1 for a, b in zip(widths_embed, widths_deep) if b < a)
    return {
        "n_cases": plotted,
        "mean_width_embed": float(np.nanmean(widths_embed)) if widths_embed else None,
        "mean_width_deep_p": float(np.nanmean(widths_deep)) if widths_deep else None,
        "frac_sharper_deep": sharper / max(plotted, 1),
        "figure": str(summary),
        "case_dir": str(chain_dir),
    }


def _make_scrub_mask(
    length: int,
    center: int,
    radius_bins: int,
    device: torch.device,
) -> torch.Tensor:
    idx = torch.arange(length, device=device)
    return ((idx >= center - radius_bins) & (idx <= center + radius_bins)).float().view(1, length, 1)


@torch.no_grad()
def run_counterfactual_rho(
    model,
    device,
    out_dir: Path,
    seq_len: int,
    n_cases: int,
    scrub_radius_sec: float,
) -> dict:
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=500)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    radius_bins = max(1, int(round(scrub_radius_sec / (60.0 / max(seq_len - 1, 1)))))
    rows = []
    plotted = 0
    panel_dir = out_dir / "counterfactual_rho"
    panel_dir.mkdir(exist_ok=True)

    for batch in loader:
        if len(rows) >= n_cases:
            break
        if float(batch["det"][0]) <= 0.5 or int(batch["s_valid"][0]) <= 0:
            continue
        x, t = batch["x"].to(device), batch["t"].to(device)
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0])

        base = forward_with_rho_override(model, x, t, None)
        rho0 = base["rho"]
        # full rho tensor for override: rebuild (B,T,1)
        _xd, x_pick, _ = model._apply_noise_cancel(x, t)
        rho_full = model.medium_net(x_pick)
        mask = _make_scrub_mask(rho_full.shape[1], s_idx, radius_bins, device)
        # damp to near-zero in window (keep softplus-ish floor)
        rho_scrub = rho_full * (1.0 - 0.95 * mask) + 1e-3 * mask
        scrub = forward_with_rho_override(model, x, t, rho_scrub)

        p0 = torch.sigmoid(base["p"][0])
        s0 = torch.sigmoid(base["s"][0])
        p1 = torch.sigmoid(scrub["p"][0])
        s1 = torch.sigmoid(scrub["s"][0])
        # local peak in GT windows
        def win_peak(prob, idx, rad=15):
            lo, hi = max(0, idx - rad), min(prob.numel(), idx + rad + 1)
            return float(prob[lo:hi].max())

        dp = win_peak(p1, p_idx) - win_peak(p0, p_idx)
        ds_ = win_peak(s1, s_idx) - win_peak(s0, s_idx)
        rows.append({"d_p_peak": dp, "d_s_peak": ds_})

        if plotted < 6:
            t_sec = t[0, :, 0].cpu().numpy()
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(t_sec, rho0[0].cpu().numpy(), label="rho base", color="C3")
            axes[0].plot(t_sec, scrub["rho"][0].cpu().numpy(), label="rho scrub S", color="C4", alpha=0.85)
            axes[0].axvline(idx_to_sec(s_idx, seq_len), color="C1", ls="--")
            axes[0].legend(fontsize=8)
            axes[0].set_ylabel("rho")
            axes[1].plot(t_sec, p0.cpu().numpy(), label="P base", color="C2")
            axes[1].plot(t_sec, p1.cpu().numpy(), label="P scrub", color="C2", ls="--")
            axes[1].legend(fontsize=8)
            axes[1].set_ylabel("P prob")
            axes[2].plot(t_sec, s0.cpu().numpy(), label="S base", color="C1")
            axes[2].plot(t_sec, s1.cpu().numpy(), label="S scrub", color="C1", ls="--")
            axes[2].axvline(idx_to_sec(s_idx, seq_len), color="C1", ls="--", alpha=0.5)
            axes[2].legend(fontsize=8)
            axes[2].set_ylabel("S prob")
            axes[2].set_xlabel("time (s)")
            for ax in axes:
                ax.grid(True, alpha=0.25)
            fig.suptitle(f"ρ scrub @ S  ΔP={dp:+.3f}  ΔS={ds_:+.3f}")
            fig.tight_layout()
            fig.savefig(panel_dir / f"scrub_{plotted:02d}.png", dpi=140)
            plt.close(fig)
            plotted += 1

    d_p = np.array([r["d_p_peak"] for r in rows])
    d_s = np.array([r["d_s_peak"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d_p, bins=12, alpha=0.6, label="ΔP peak @P")
    ax.hist(d_s, bins=12, alpha=0.6, label="ΔS peak @S")
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.set_xlabel("probability delta after scrubbing ρ near S")
    ax.set_ylabel("count")
    ax.set_title("Counterfactual ρ intervention")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = out_dir / "counterfactual_rho_delta_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    return {
        "n_cases": len(rows),
        "mean_d_p_peak": float(d_p.mean()) if len(d_p) else None,
        "mean_d_s_peak": float(d_s.mean()) if len(d_s) else None,
        "frac_s_drops": float((d_s < -0.02).mean()) if len(d_s) else None,
        "figure": str(hist_path),
        "case_dir": str(panel_dir),
        "scrub_radius_sec": scrub_radius_sec,
    }


@torch.no_grad()
def run_anomaly_attribution(
    model,
    device,
    out_dir: Path,
    seq_len: int,
    n_cases: int,
) -> dict:
    """Noise traces where max P-prob is high → inspect kernel row peak lag."""
    ds = STEADPickingDataset(
        "test", seq_len=seq_len, max_event_traces=20, max_noise_traces=1500
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    anom_dir = out_dir / "anomaly_krow"
    anom_dir.mkdir(exist_ok=True)
    records = []
    plotted = 0

    for batch in loader:
        if plotted >= n_cases:
            break
        is_noise = float(batch["det"][0]) < 0.5
        if not is_noise:
            continue
        x, t = batch["x"].to(device), batch["t"].to(device)
        out = model(x, t)
        p_prob = torch.sigmoid(out["p"][0]).cpu().numpy()
        peak_p = float(p_prob.max())
        if peak_p < 0.40:
            continue
        peak_idx = int(p_prob.argmax())
        expl = model.forward_explain(
            x, t, include_kernel_row=True, kernel_row_idx=peak_idx, kernel_branch="p"
        )
        k_row = expl["kernel_contrib"][0].cpu().numpy()
        k_peak = int(np.argmax(k_row))
        lag = peak_idx - k_peak
        records.append({"peak_p": peak_p, "lag": lag, "k_peak": k_peak, "p_peak": peak_idx})

        t_sec = t[0, :, 0].cpu().numpy()
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t_sec, x[0, :, 2].cpu().numpy(), color="0.3", lw=0.8)
        axes[0].set_ylabel("Z (noise)")
        axes[0].set_title(f"false-P candidate peak_p={peak_p:.2f}")
        axes[1].plot(t_sec, p_prob, color="C2")
        axes[1].axvline(idx_to_sec(peak_idx, seq_len), color="C2", ls="--")
        axes[1].set_ylabel("P prob")
        axes[2].plot(t_sec, k_row, color="C4")
        axes[2].axvline(idx_to_sec(k_peak, seq_len), color="C4", ls="--")
        axes[2].set_ylabel("|K| row")
        axes[2].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(anom_dir / f"anom_{plotted:02d}.png", dpi=140)
        plt.close(fig)
        plotted += 1

    lags = [r["lag"] for r in records]
    return {
        "n_cases": plotted,
        "mean_false_p_peak": float(np.mean([r["peak_p"] for r in records])) if records else None,
        "mean_k_to_p_lag_bins": float(np.mean(lags)) if lags else None,
        "case_dir": str(anom_dir),
        "note": "Kernel peaks far from high P-prob on noise suggest perception of non-physical patterns.",
    }


def copy_docs(out_dir: Path) -> list[str]:
    dest = Path("docs/figures/probing")
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in [
        "causal_chain_focusing_scatter.png",
        "counterfactual_rho_delta_hist.png",
    ]:
        src = out_dir / pat
        if src.is_file():
            shutil.copy2(src, dest / pat)
            copied.append(str(dest / pat))
    # copy a few case panels
    for sub in ["causal_chain", "counterpart"]:
        pass
    for sub in ["causal_chain", "counterfactual_rho", "anomaly_krow"]:
        sdir = out_dir / sub
        if not sdir.is_dir():
            continue
        dsub = dest / sub
        dsub.mkdir(exist_ok=True)
        for i, f in enumerate(sorted(sdir.glob("*.png"))[:4]):
            shutil.copy2(f, dsub / f.name)
            copied.append(str(dsub / f.name))
    return copied


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probing] load {args.checkpoint}", flush=True)
    model, meta = load_model(Path(args.checkpoint), device, bypass_noise_cancel=True)
    report: dict = {
        "checkpoint": args.checkpoint,
        "n_params": sum(p.numel() for p in model.parameters()),
        "principle": (meta or {}).get("principle"),
        "multi_scale": (meta or {}).get("multi_scale"),
    }

    print("[probing] (1) causal-chain tracking...", flush=True)
    report["causal_chain"] = run_causal_chain(
        model, device, out_dir, args.seq_len, args.n_chain
    )

    print("[probing] (2) counterfactual rho scrubbing...", flush=True)
    report["counterfactual_rho"] = run_counterfactual_rho(
        model, device, out_dir, args.seq_len, args.n_counterfactual, args.scrub_radius_sec
    )

    print("[probing] (3) anomaly attribution...", flush=True)
    report["anomaly"] = run_anomaly_attribution(
        model, device, out_dir, args.seq_len, args.n_anomaly
    )

    if args.copy_to_docs:
        report["docs_copied"] = copy_docs(out_dir)

    (out_dir / "probing_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# HNF Probing Report",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        "",
        "## Causal-chain focusing",
        f"- n: {report['causal_chain'].get('n_cases')}",
        f"- mean width embed → deep P: "
        f"{report['causal_chain'].get('mean_width_embed')} → "
        f"{report['causal_chain'].get('mean_width_deep_p')}",
        f"- frac sharper deep: {report['causal_chain'].get('frac_sharper_deep')}",
        "",
        "## Counterfactual ρ scrub (near S)",
        f"- n: {report['counterfactual_rho'].get('n_cases')}",
        f"- mean ΔP / ΔS: {report['counterfactual_rho'].get('mean_d_p_peak')} / "
        f"{report['counterfactual_rho'].get('mean_d_s_peak')}",
        f"- frac S drops (>0.02): {report['counterfactual_rho'].get('frac_s_drops')}",
        "",
        "## Anomaly K-row on noise false-P",
        f"- n: {report['anomaly'].get('n_cases')}",
        f"- mean false P peak: {report['anomaly'].get('mean_false_p_peak')}",
        "",
    ]
    (out_dir / "probing_report.md").write_text("\n".join(md))
    print(json.dumps({
        "causal_chain": report["causal_chain"],
        "counterfactual_rho": {
            k: report["counterfactual_rho"].get(k)
            for k in ["n_cases", "mean_d_p_peak", "mean_d_s_peak", "frac_s_drops"]
        },
        "anomaly_n": report["anomaly"].get("n_cases"),
        "out": str(out_dir),
    }, indent=2))
    print(f"[probing] -> {out_dir}")


if __name__ == "__main__":
    main()
