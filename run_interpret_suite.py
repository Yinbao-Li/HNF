#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpretability proof package for HNF (picking + Zhizi bridge).

Sections:
  A) Kernel physics: obliquity χ(θ), Huygens vs Huygens–Fresnel |K| difference
  B) Picking explain: rho / envelope / kernel row contribution (run20)
  C) Latent bridge panels + rho-vs-distance (macro head)
  D) Principle ablation: run20 vs Fresnel picking metrics
  E) Inversion init→refine scatter (synthetic)

Usage:
  python run_interpret_suite.py --device cuda
  python run_interpret_suite.py --device cuda --copy-to-docs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model as load_picking_ckpt
from eval_stead_picking import evaluate_checkpoint
from hnf.kernel import HuygensKernel
from hnf.picking_metrics import idx_to_sec
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.zhizi_inversion_bridge import ZhiziInversionBridge, load_physics_head_state
from hnf.inversion_1d import default_synth_model


RUN20_REF = {
    "det_f1": 0.9940792470016699,
    "p_f1": 0.9593805429066354,
    "s_f1": 0.9492358762370036,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF interpretability suite")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge_macro/best_physics_head.pt")
    p.add_argument("--fresnel-checkpoint", default="outputs/huygens_fresnel/picking_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/interpret_suite")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--n-latent", type=int, default=6)
    p.add_argument("--n-kernel-rows", type=int, default=4)
    p.add_argument("--copy-to-docs", action="store_true")
    return p.parse_args()


def plot_obliquity_and_kernel_diff(out_dir: Path, device: torch.device) -> dict:
    """Fresnel obliquity χ(lag) and |K_hf|−|K_h| on a uniform time grid."""
    n = 120
    t = torch.linspace(0, 15.0, n, device=device).view(1, n, 1)
    x = torch.zeros(1, n, 4, device=device)

    common = dict(
        gamma=0.5,
        omega=0.3,
        causal=True,
        wave_speed=6.0,
        distance_mode="time",
        local_window_sec=15.0,
        obliquity_scale=1.0,
    )
    k_h = HuygensKernel(principle="huygens", **common).to(device)
    k_f = HuygensKernel(principle="huygens_fresnel", **common).to(device)

    with torch.no_grad():
        r = k_h.resolve_distance(x, t=t)
        chi = k_f._fresnel_obliquity(r, t=t, x=x)[0].cpu().numpy()
        kh = torch.abs(k_h(x, t=t, return_complex=True))[0].cpu().numpy()
        kf = torch.abs(k_f(x, t=t, return_complex=True))[0].cpu().numpy()
        diff = kf - kh

    lags = np.linspace(0, 15.0, n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    im0 = axes[0].imshow(chi, aspect="auto", origin="lower", cmap="magma",
                         extent=[0, 15, 0, 15])
    axes[0].set_title("Fresnel obliquity χ(θ)")
    axes[0].set_xlabel("source lag (s)")
    axes[0].set_ylabel("receiver time (s)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.log10(kh + 1e-8), aspect="auto", origin="lower", cmap="viridis",
                         extent=[0, 15, 0, 15])
    axes[1].set_title("|K| Huygens (log10)")
    axes[1].set_xlabel("source lag (s)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                         extent=[0, 15, 0, 15],
                         vmin=-np.percentile(np.abs(diff), 99),
                         vmax=np.percentile(np.abs(diff), 99))
    axes[2].set_title("|K_Fresnel| − |K_Huygens|")
    axes[2].set_xlabel("source lag (s)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.tight_layout()
    p_main = out_dir / "kernel_obliquity_diff.png"
    fig.savefig(p_main, dpi=150)
    plt.close(fig)

    # 1D slice at representative receiver index
    row = n // 2
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lags, chi[row], label="χ along causal row", color="C3")
    ax2 = ax.twinx()
    ax2.plot(lags, kh[row], label="|K| Huygens", color="C0", alpha=0.7)
    ax2.plot(lags, kf[row], label="|K| Fresnel", color="C1", alpha=0.7, ls="--")
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("obliquity χ")
    ax2.set_ylabel("|K|")
    ax.set_title(f"Causal kernel row t≈{lags[row]:.1f}s")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_slice = out_dir / "kernel_row_slice.png"
    fig.savefig(p_slice, dpi=150)
    plt.close(fig)

    return {
        "chi_mean_causal": float(chi[chi > 0.01].mean()) if (chi > 0.01).any() else None,
        "mean_abs_kernel_diff": float(np.mean(np.abs(diff))),
        "figures": [str(p_main), str(p_slice)],
    }


def plot_picking_principle_compare(out_dir: Path, run20_ckpt: Path, fresnel_ckpt: Path, device: str) -> dict:
    """Bar chart: run20 vs Fresnel test metrics."""
    rows = []
    pick_json = Path("outputs/huygens_fresnel/pick_compare.json")
    if pick_json.is_file():
        data = json.loads(pick_json.read_text())
        rows = [
            ("run20 (Huygens)", dict(RUN20_REF)),
            ("Fresnel", data.get("fresnel_summary", data.get("fresnel_test", {}))),
        ]
    else:
        for name, ckpt in [("run20", run20_ckpt), ("fresnel", fresnel_ckpt)]:
            if not ckpt.is_file():
                continue
            m = evaluate_checkpoint(
                str(ckpt),
                {"seq_len": 800},
                post_process_p_before_s=True,
                device=device,
            )
            rows.append((name, m))

    labels = ["det_f1", "p_f1", "s_f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    w = 0.35
    for i, (name, m) in enumerate(rows):
        vals = [float(m.get(k, 0)) for k in labels]
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("F1")
    ax.set_title("Picking: Huygens (run20) vs Huygens–Fresnel")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "picking_principle_compare.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    delta = {}
    if len(rows) == 2:
        for k in labels:
            delta[k] = float(rows[1][1].get(k, 0)) - float(rows[0][1].get(k, 0))
    return {"metrics": {n: m for n, m in rows}, "delta_fresnel_minus_run20": delta, "figure": str(p)}


def run_kernel_contrib_panels(
    model,
    device: torch.device,
    out_dir: Path,
    seq_len: int,
    n_cases: int,
) -> dict:
    """forward_explain with kernel row for P branch at GT P index."""
    from hnf.picking_model import STEADHNFPickingModel

    assert isinstance(model, STEADHNFPickingModel)
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=300)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    cases = []
    plotted = 0

    for batch in loader:
        if plotted >= n_cases:
            break
        if float(batch["det"][0]) <= 0.5 or batch["p_valid"][0] <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0]) if batch["s_valid"][0] > 0 else -1
        with torch.no_grad():
            out = model.forward_explain(
                x, t, include_kernel_row=True, kernel_row_idx=p_idx, kernel_branch="p"
            )
        t_sec = t[0, :, 0].cpu().numpy()
        z = x[0, :, 2].cpu().numpy()
        rho = out["rho"][0].cpu().numpy()
        p_prob = torch.sigmoid(out["p"][0]).cpu().numpy()
        s_prob = torch.sigmoid(out["s"][0]).cpu().numpy()
        k_row = out["kernel_contrib"][0].cpu().numpy()
        gt_p = idx_to_sec(p_idx, seq_len)
        gt_s = idx_to_sec(s_idx, seq_len) if s_idx >= 0 else None

        fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t_sec, z, color="0.2", lw=0.8)
        axes[0].set_ylabel("Z")
        axes[0].set_title(batch["trace_name"][0] if isinstance(batch["trace_name"], (list, tuple)) else str(batch["trace_name"]))
        axes[1].plot(t_sec, rho, color="C3")
        axes[1].set_ylabel("rho(t)")
        axes[2].plot(t_sec, out["p_envelope"][0].cpu().numpy(), color="C0", label="P env")
        axes[2].plot(t_sec, out["s_envelope"][0].cpu().numpy(), color="C1", label="S env", alpha=0.8)
        axes[2].legend(fontsize=8)
        axes[2].set_ylabel("envelope")
        axes[3].plot(t_sec, k_row, color="C4")
        axes[3].set_ylabel("|K| row@P")
        axes[3].set_title("P-branch kernel contribution to GT P index")
        axes[4].plot(t_sec, p_prob, color="C2", label="P")
        axes[4].plot(t_sec, s_prob, color="C1", label="S")
        axes[4].axvline(gt_p, color="C2", ls="--", alpha=0.7)
        if gt_s is not None:
            axes[4].axvline(gt_s, color="C1", ls="--", alpha=0.7)
        axes[4].set_ylabel("pick prob")
        axes[4].set_xlabel("time (s)")
        axes[4].legend(fontsize=8)
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fp = out_dir / f"kernel_contrib_{plotted:02d}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)

        # rho in S window vs noise
        s0 = max(0, s_idx - 20)
        s1 = min(len(rho), s_idx + 40) if s_idx >= 0 else len(rho) // 2
        n0, n1 = 0, min(80, len(rho) // 4)
        cases.append({
            "trace": str(batch["trace_name"][0]),
            "rho_mean_noise": float(np.mean(rho[n0:n1])),
            "rho_mean_s_window": float(np.mean(rho[s0:s1])) if s1 > s0 else None,
            "rho_ratio_s_over_noise": float(np.mean(rho[s0:s1]) / max(np.mean(rho[n0:n1]), 1e-6)) if s1 > s0 else None,
            "figure": str(fp),
        })
        plotted += 1

    if cases:
        ratios = [c["rho_ratio_s_over_noise"] for c in cases if c["rho_ratio_s_over_noise"] is not None]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(ratios, bins=min(8, len(ratios)), color="C3", alpha=0.85)
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("mean rho(S window) / mean rho(noise)")
        ax.set_ylabel("count")
        ax.set_title(f"Latent rho aligns with S energy (n={len(ratios)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "rho_s_over_noise_hist.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
    else:
        p = None

    return {"n_cases": len(cases), "cases": cases, "rho_hist": str(p) if p else None}


def run_bridge_latent_panels(
    args,
    device: torch.device,
    out_dir: Path,
) -> dict:
    backbone, ckpt_args = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    base = default_synth_model(device)
    state = torch.load(args.physics_head, map_location=device, weights_only=False)
    geo_condition = bool(state.get("geo_condition", False)) or bool(
        (state.get("args") or {}).get("geo_condition", False)
    )
    bridge = ZhiziInversionBridge(
        backbone=backbone,
        n_layers=base.n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=600,
        head_mode="macro",
        geo_condition=geo_condition,
    ).to(device)
    load_physics_head_state(bridge.physics_head, state["physics_head"])
    bridge.eval()

    ds = STEADPickingDataset("test", seq_len=args.seq_len, max_event_traces=200)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows = []
    plotted = 0
    for batch in loader:
        if plotted >= args.n_latent:
            break
        if float(batch["det"][0]) <= 0.5 or batch["p_valid"][0] <= 0:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        with torch.no_grad():
            feat = bridge.extract_station_features(x, t, include_picks=True)
        rho = feat["rho"][0].detach().cpu().numpy()
        hr = feat["h_real"][0].detach().cpu().numpy()
        env = np.sqrt((hr ** 2).sum(axis=-1) + 1e-8)
        p_prob = 1.0 / (1.0 + np.exp(-feat["p_logits"][0].detach().cpu().numpy()))
        s_prob = 1.0 / (1.0 + np.exp(-feat["s_logits"][0].detach().cpu().numpy()))
        x_np = x[0].detach().cpu().numpy()
        seq = min(len(rho), len(env), x_np.shape[0])
        t_use = np.linspace(0, 60, seq)
        dist = float(batch["source_distance_km"][0])
        gt_p = idx_to_sec(int(batch["p_idx"][0]), x.shape[1])
        gt_s = idx_to_sec(int(batch["s_idx"][0]), x.shape[1]) if batch["s_valid"][0] > 0 else None

        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t_use, x_np[:seq, 2], color="0.2", lw=0.8)
        axes[0].set_ylabel("Z")
        axes[0].set_title(f"dist={dist:.1f}km  {batch['trace_name'][0]}")
        axes[1].plot(t_use, rho[:seq], color="C3")
        axes[1].set_ylabel("rho(t)")
        axes[2].plot(t_use, env[:seq], color="C0")
        axes[2].set_ylabel("envelope")
        axes[3].plot(t_use, p_prob[:seq], color="C2", label="P")
        axes[3].plot(t_use, s_prob[:seq], color="C1", label="S")
        axes[3].axvline(gt_p, color="C2", ls="--", alpha=0.7)
        if gt_s is not None:
            axes[3].axvline(gt_s, color="C1", ls="--", alpha=0.7)
        axes[3].legend(fontsize=8)
        axes[3].set_xlabel("time (s)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fp = out_dir / f"bridge_latent_{plotted:02d}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        rows.append({"distance_km": dist, "rho_mean": float(np.mean(rho[:seq])), "figure": str(fp)})
        plotted += 1

    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter([r["distance_km"] for r in rows], [r["rho_mean"] for r in rows], c="C3")
        ax.set_xlabel("source_distance_km")
        ax.set_ylabel("mean rho(t)")
        ax.set_title("Bridge latent rho vs distance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "bridge_rho_vs_distance.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
    return {"n": len(rows), "cases": rows}


def plot_inversion_init_refine(out_dir: Path) -> dict:
    """Copy/summarize init vs wave scatter from proof outputs if present."""
    proof = Path("outputs/proof_suite/synth_full_compare.json")
    fresnel = Path("outputs/huygens_fresnel/proof_suite/synth_full_compare.json")
    fig_path = None
    summary = {}
    if proof.is_file():
        data = json.loads(proof.read_text())
        per = data.get("per_event", [])
        if per:
            zh_i = [r["zhizi_init"] for r in per]
            zh_w = [r["zhizi_wave"] for r in per]
            pe_w = [r["perturb_wave"] for r in per]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].scatter(zh_i, zh_w, alpha=0.7, c="C0")
            lim = max(max(zh_i + zh_w), 0.1)
            axes[0].plot([0, lim], [0, lim], "k--", lw=1)
            axes[0].set_xlabel("zhizi init VpRMSE")
            axes[0].set_ylabel("zhizi wave VpRMSE")
            axes[0].set_title("Init → wave refine (run20)")
            axes[1].hist(np.array(zh_w) - np.array(pe_w), bins=10, color="C0", alpha=0.8)
            axes[1].axvline(0, color="k", ls="--")
            axes[1].set_xlabel("zhizi_wave − perturb_wave")
            axes[1].set_title("Route A2 paired delta")
            for ax in axes:
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = out_dir / "inversion_init_refine.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            summary["run20"] = {
                "mean_zhizi_wave": float(np.mean(zh_w)),
                "mean_perturb_wave": float(np.mean(pe_w)),
                "win_frac": float(np.mean([a < b for a, b in zip(zh_w, pe_w)])),
            }
    if fresnel.is_file():
        fd = json.loads(fresnel.read_text())
        summary["fresnel"] = {
            "mean_zhizi_wave": fd.get("means", {}).get("zhizi_wave"),
            "win_frac": fd.get("zhizi_wave_better_than_perturb_frac"),
        }
    return {"figure": str(fig_path) if fig_path else None, "summary": summary}


def copy_docs(out_dir: Path) -> list[str]:
    doc_dir = Path("docs/figures/interpret")
    doc_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(out_dir.glob("*.png")):
        dst = doc_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    for sub in ["kernel_contrib", "bridge_latent"]:
        sd = out_dir / sub
        if not sd.is_dir():
            continue
        (doc_dir / sub).mkdir(exist_ok=True)
        for src in sorted(sd.glob("*.png")):
            dst = doc_dir / sub / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    kdir = out_dir / "kernel_contrib"
    bdir = out_dir / "bridge_latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    kdir.mkdir(exist_ok=True)
    bdir.mkdir(exist_ok=True)

    report = {"checkpoint": args.checkpoint, "physics_head": args.physics_head}

    print("[interpret] kernel obliquity + diff...", flush=True)
    report["kernel_physics"] = plot_obliquity_and_kernel_diff(out_dir, device)

    print("[interpret] picking principle compare...", flush=True)
    report["picking_ablation"] = plot_picking_principle_compare(
        out_dir, Path(args.checkpoint), Path(args.fresnel_checkpoint), args.device
    )

    print("[interpret] kernel contribution panels (run20)...", flush=True)
    model, _ = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    report["kernel_contrib"] = run_kernel_contrib_panels(
        model, device, kdir, args.seq_len, args.n_kernel_rows
    )

    print("[interpret] bridge latent panels...", flush=True)
    report["bridge_latent"] = run_bridge_latent_panels(args, device, bdir)

    print("[interpret] inversion init→refine...", flush=True)
    report["inversion"] = plot_inversion_init_refine(out_dir)

    # Merge fresnel invert compare if present
    inv_cmp = Path("outputs/huygens_fresnel/invert_compare.json")
    if inv_cmp.is_file():
        report["fresnel_inversion_ablation"] = json.loads(inv_cmp.read_text())

    report["interpretation_notes"] = {
        "rho": "Soft latent weight; higher in energetic / S intervals — not crustal density.",
        "obliquity": "Fresnel χ suppresses off-axis secondary sources; changes |K| mainly at longer lags.",
        "kernel_row": "Causal light-cone row at GT P shows which past samples contribute to pick.",
        "fresnel_verdict": "Fresnel picking: det +0.002, P/S −0.034/−0.022 vs run20; inversion still PASS but marginal.",
    }

    (out_dir / "interpret_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# HNF Interpretability Report",
        "",
        "## Kernel physics",
        f"- Mean |K_Fresnel − K_Huygens|: {report['kernel_physics'].get('mean_abs_kernel_diff')}",
        f"- See `kernel_obliquity_diff.png`, `kernel_row_slice.png`",
        "",
        "## Picking (run20 vs Fresnel)",
        f"- Delta: {report['picking_ablation'].get('delta_fresnel_minus_run20')}",
        "",
        "## Latent rho",
        f"- S-window / noise rho ratio cases: {report['kernel_contrib'].get('n_cases')}",
        "",
        "## Inversion",
        f"- {report['inversion'].get('summary')}",
        "",
        "Run: `python run_interpret_suite.py --device cuda --copy-to-docs`",
    ]
    (out_dir / "interpret_report.md").write_text("\n".join(md))

    if args.copy_to_docs:
        copied = copy_docs(out_dir)
        report["docs_copied"] = copied
        (out_dir / "interpret_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "kernel": report["kernel_physics"],
        "picking_delta": report["picking_ablation"].get("delta_fresnel_minus_run20"),
        "rho_cases": report["kernel_contrib"].get("n_cases"),
        "inversion": report["inversion"].get("summary"),
        "out": str(out_dir),
    }, indent=2))
    print(f"[interpret] -> {out_dir}")


if __name__ == "__main__":
    main()
