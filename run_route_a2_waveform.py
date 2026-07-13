#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route A2: Zhizi initialization + differentiable waveform inversion.

Minimal physical closed loop:
  waveform -> Zhizi init vp/vs -> differentiable waveform forward -> gradient update m

Compares Zhizi init against perturb init under the same waveform inversion engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.acoustic_fwi_1d import DirectWaveForward, invert_acoustic_fwi
from hnf.inversion_1d import LayeredEarth1D, default_synth_model, model_rmse
from hnf.inv_plot import perturb_initial
from hnf.ray_paths import direct_ray_path
from hnf.physics_decoder import PhysicsDecoder
from hnf.zhizi_inversion_dataset import ZhiziInversionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route A2: Zhizi init + waveform inversion")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge_residual/best_physics_head.pt")
    p.add_argument("--head-mode", choices=["residual", "macro"], default="residual")
    p.add_argument("--output-dir", default="outputs/route_a2_waveform")
    p.add_argument("--n-test", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--fwi-steps", type=int, default=80)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(len(rows), 1)


def plot_paths(out_path: Path, true_earth: LayeredEarth1D, zhizi_earth: LayeredEarth1D, refined_earth: LayeredEarth1D, source_depth: float, distances: torch.Tensor) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    models = [
        ("True", true_earth, "C2"),
        ("Zhizi init", zhizi_earth, "C0"),
        ("Zhizi + waveform", refined_earth, "C3"),
    ]
    for ax, (title, earth, color) in zip(axes, models):
        for phase, ls in [("P", "-"), ("S", "--")]:
            for d in distances[: min(4, distances.numel())]:
                x, z = direct_ray_path(earth, phase, source_depth, float(d.item()))
                ax.plot(x.detach().cpu().numpy(), z.detach().cpu().numpy(), ls, color=color, alpha=0.75)
        for z in earth.depths.detach().cpu().numpy():
            ax.axhline(z, color="0.85", lw=0.8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("z (km)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    head_ckpt = Path(args.physics_head)
    backbone, ckpt_args = load_model(ckpt, device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    n_layers = default_synth_model(device).n_layers
    bridge = PhysicsDecoder(
        backbone=backbone,
        n_layers=n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=args.infer_seq_len,
        head_mode=args.head_mode,
    ).to(device)
    state = torch.load(head_ckpt, map_location=device, weights_only=False)
    bridge.physics_head.load_state_dict(state["physics_head"])
    bridge.eval()

    ds = ZhiziInversionDataset(n_samples=args.n_test, seq_len=args.infer_seq_len, seed=args.seed, device=device)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows: list[dict] = []
    example_saved = False

    for idx, batch in enumerate(loader):
        x = batch["x"][0].to(device)
        t = batch["t"].to(device)
        true_vp = batch["true_vp"][0].to(device)
        true_vs = batch["true_vs"][0].to(device)
        true_q = batch["true_q"][0].to(device)
        depths = batch["depths"][0].to(device)
        distances = batch["distances"][0].to(device)
        source_depth = float(batch["source_depth"][0])
        true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=true_q)

        engine = DirectWaveForward(device=device, nt=x.shape[0], dt=60.0 / max(x.shape[0] - 1, 1))
        observed = engine.simulate(true_earth, source_depth, distances)

        base = default_synth_model(device)
        vp_pert, vs_pert, q_init = perturb_initial(base.vp, base.vs, base.q, seed=args.seed + idx * 9973, q_scale=1.0)

        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True)
        zhizi_earth = bridge.physics_head.earth(out, depths, true_q)

        zh_model, zh_hist, _ = invert_acoustic_fwi(
            depths, zhizi_earth.vp, zhizi_earth.vs, q_init,
            true_earth, source_depth, distances, observed,
            steps=args.fwi_steps, verbose=False,
        )
        pe_model, pe_hist, _ = invert_acoustic_fwi(
            depths, vp_pert, vs_pert, q_init,
            true_earth, source_depth, distances, observed,
            steps=args.fwi_steps, verbose=False,
        )

        zh_init_rmse = model_rmse(true_earth, zhizi_earth)
        pe_init_rmse = model_rmse(true_earth, LayeredEarth1D(depths=depths, vp=vp_pert, vs=vs_pert, q=q_init))
        zh_ref_rmse = model_rmse(true_earth, zh_model.earth)
        pe_ref_rmse = model_rmse(true_earth, pe_model.earth)

        row = {
            "idx": idx,
            "zhizi_init_vp_rmse": zh_init_rmse["vp_rmse"],
            "perturb_init_vp_rmse": pe_init_rmse["vp_rmse"],
            "zhizi_wave_vp_rmse": zh_ref_rmse["vp_rmse"],
            "perturb_wave_vp_rmse": pe_ref_rmse["vp_rmse"],
            "zhizi_wave_loss": float(zh_hist[-1]["waveform"]),
            "perturb_wave_loss": float(pe_hist[-1]["waveform"]),
        }
        rows.append(row)
        print(
            f"[{idx+1}/{args.n_test}] init z={row['zhizi_init_vp_rmse']:.3f} p={row['perturb_init_vp_rmse']:.3f} | "
            f"wave z={row['zhizi_wave_vp_rmse']:.3f} p={row['perturb_wave_vp_rmse']:.3f}",
            flush=True,
        )

        if not example_saved:
            plot_paths(out_dir / "example_ray_paths.png", true_earth, zhizi_earth, zh_model.earth, source_depth, distances)
            example = {
                "true_vp": true_earth.vp.cpu().tolist(),
                "zhizi_init_vp": zhizi_earth.vp.cpu().tolist(),
                "zhizi_wave_vp": zh_model.earth.vp.cpu().tolist(),
                "perturb_init_vp": vp_pert.cpu().tolist(),
                "perturb_wave_vp": pe_model.earth.vp.cpu().tolist(),
            }
            (out_dir / "example_models.json").write_text(json.dumps(example, indent=2))
            example_saved = True

    zh_better_frac = sum(1 for r in rows if r["zhizi_wave_vp_rmse"] < r["perturb_wave_vp_rmse"]) / max(len(rows), 1)
    summary = {
        "route": "A2",
        "question": "智子速度作为可微波形反演初值是否更有物理意义？",
        "n_test": len(rows),
        "fwi_steps": args.fwi_steps,
        "init": {
            "zhizi_vp_rmse_mean": mean(rows, "zhizi_init_vp_rmse"),
            "perturb_vp_rmse_mean": mean(rows, "perturb_init_vp_rmse"),
        },
        "waveform_refined": {
            "zhizi_vp_rmse_mean": mean(rows, "zhizi_wave_vp_rmse"),
            "perturb_vp_rmse_mean": mean(rows, "perturb_wave_vp_rmse"),
            "zhizi_wave_loss_mean": mean(rows, "zhizi_wave_loss"),
            "perturb_wave_loss_mean": mean(rows, "perturb_wave_loss"),
            "zhizi_better_frac": zh_better_frac,
        },
        "per_event": rows,
    }
    (out_dir / "route_a2_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_event"}, indent=2))
    print(f"[route-a2] -> {out_dir}")


if __name__ == "__main__":
    main()
