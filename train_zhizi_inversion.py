#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Zhizi inversion bridge (frozen backbone + Physics Head).

Stage 1: synthetic 1D layered Earth, travel-time physics loss + soft latent priors.
Does NOT update picking/det/P/S heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.acoustic_fwi_1d import DirectWaveForward, unrolled_waveform_refine
from hnf.inversion_1d import LayeredEarth1D, default_synth_model
from hnf.inv_plot import perturb_initial
from hnf.zhizi_inversion_bridge import ZhiziInversionBridge
from hnf.zhizi_inversion_dataset import ZhiziInversionDataset
from hnf.zhizi_inversion_loss import zhizi_inversion_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Zhizi inversion Physics Head")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/zhizi_inversion_bridge")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=24)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seq-len", type=int, default=600)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--vp-sup-weight", type=float, default=0.05, help="Synthetic vp supervision")
    p.add_argument("--waveform-weight", type=float, default=0.0, help="Waveform loss on Zhizi initial model")
    p.add_argument("--unrolled-weight", type=float, default=0.0, help="Loss after short differentiable waveform refinement")
    p.add_argument("--unrolled-steps", type=int, default=3)
    p.add_argument("--unrolled-step-size", type=float, default=0.05)
    p.add_argument("--pairwise-weight", type=float, default=0.0, help="Make Zhizi beat perturb baseline after unrolled refine")
    p.add_argument("--pairwise-margin", type=float, default=0.0)
    p.add_argument("--head-mode", choices=["residual", "macro"], default="residual")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(
    bridge: ZhiziInversionBridge,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    vp_sup_weight: float,
    waveform_weight: float,
    unrolled_weight: float,
    unrolled_steps: int,
    unrolled_step_size: float,
    pairwise_weight: float,
    pairwise_margin: float,
    pairwise_vp_init: torch.Tensor,
    pairwise_vs_init: torch.Tensor,
    pairwise_q_init: torch.Tensor,
) -> dict[str, float]:
    train = optimizer is not None
    bridge.train(train)
    bridge.backbone.eval()

    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        obs_tp = batch["obs_tp"].to(device)
        obs_ts = batch["obs_ts"].to(device)
        depths = batch["depths"][0].to(device)
        true_vp = batch["true_vp"][0].to(device)
        true_vs = batch["true_vs"][0].to(device)
        true_q = batch["true_q"][0].to(device)
        distances = batch["distances"][0].to(device)
        source_depth = float(batch["source_depth"][0])

        if train:
            optimizer.zero_grad()

        output, rho_layers = bridge.forward_event(x[0], t, include_picks=True)
        loss, metrics = zhizi_inversion_loss(
            output,
            depths=depths,
            q=true_q,
            source_depth=source_depth,
            receiver_distances=distances,
            obs_tp=obs_tp[0],
            obs_ts=obs_ts[0],
            rho_layers=rho_layers[0],
            true_vp=true_vp,
            true_vs=true_vs,
            vp_sup_weight=vp_sup_weight if train else 0.0,
        )

        if waveform_weight > 0:
            dt = float(t[1, 0] - t[0, 0]) if t.shape[0] > 1 else 0.01
            engine = DirectWaveForward(device=device, nt=t.shape[0], dt=dt)
            pred_earth = bridge.physics_head.earth(output, depths, true_q)
            true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=true_q)
            pred_wf = engine.simulate(pred_earth, source_depth, distances)
            obs_wf = engine.simulate(true_earth, source_depth, distances)
            loss_wf = torch.mean((pred_wf - obs_wf) ** 2)
            loss = loss + waveform_weight * loss_wf
            metrics["loss_waveform"] = float(loss_wf.detach())

        if unrolled_weight > 0:
            dt = float(t[1, 0] - t[0, 0]) if t.shape[0] > 1 else 0.01
            engine = DirectWaveForward(device=device, nt=t.shape[0], dt=dt)
            true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=true_q)
            obs_wf = engine.simulate(true_earth, source_depth, distances)
            refined_earth, refined_metrics = unrolled_waveform_refine(
                depths=depths,
                vp_init=output.vp[0],
                vs_init=output.vs[0],
                q=true_q,
                source_depth=source_depth,
                receiver_distances=distances,
                observed=obs_wf,
                steps=unrolled_steps,
                step_size=unrolled_step_size,
                dt=dt,
            )
            loss_unrolled = torch.mean((refined_earth.vp - true_vp) ** 2)
            loss = loss + unrolled_weight * loss_unrolled
            metrics["loss_unrolled_vp"] = float(loss_unrolled.detach())
            metrics["loss_unrolled_waveform"] = float(refined_metrics["waveform"].detach())

        if pairwise_weight > 0:
            dt = float(t[1, 0] - t[0, 0]) if t.shape[0] > 1 else 0.01
            engine = DirectWaveForward(device=device, nt=t.shape[0], dt=dt)
            true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=true_q)
            obs_wf = engine.simulate(true_earth, source_depth, distances)
            zh_refined, _ = unrolled_waveform_refine(
                depths=depths,
                vp_init=output.vp[0],
                vs_init=output.vs[0],
                q=true_q,
                source_depth=source_depth,
                receiver_distances=distances,
                observed=obs_wf,
                steps=unrolled_steps,
                step_size=unrolled_step_size,
                dt=dt,
            )
            pe_refined, _ = unrolled_waveform_refine(
                depths=depths,
                vp_init=pairwise_vp_init,
                vs_init=pairwise_vs_init,
                q=pairwise_q_init,
                source_depth=source_depth,
                receiver_distances=distances,
                observed=obs_wf,
                steps=unrolled_steps,
                step_size=unrolled_step_size,
                dt=dt,
            )
            zh_err = torch.mean((zh_refined.vp - true_vp) ** 2)
            pe_err = torch.mean((pe_refined.vp.detach() - true_vp) ** 2)
            pairwise_loss = torch.relu(zh_err - pe_err + pairwise_margin)
            loss = loss + pairwise_weight * pairwise_loss
            metrics["loss_pairwise"] = float(pairwise_loss.detach())
            metrics["pairwise_zh_err"] = float(zh_err.detach())
            metrics["pairwise_pe_err"] = float(pe_err.detach())

        if train:
            loss.backward()
            optimizer.step()

        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[zhizi-inv] loading backbone {ckpt}", flush=True)
    backbone, ckpt_args = load_model(ckpt, device, bypass_noise_cancel=True)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    n_layers = default_synth_model(device).n_layers

    bridge = ZhiziInversionBridge(
        backbone=backbone,
        n_layers=n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=args.infer_seq_len,
        head_mode=args.head_mode,
    ).to(device)

    print(
        f"[zhizi-inv] trainable params={bridge.trainable_parameter_count()} "
        f"total={bridge.total_parameter_count()}",
        flush=True,
    )

    train_ds = ZhiziInversionDataset(
        n_samples=args.n_train, seq_len=args.seq_len, seed=args.seed, device=device
    )
    val_ds = ZhiziInversionDataset(
        n_samples=args.n_val, seq_len=args.seq_len, seed=args.seed + 999, device=device
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    base = default_synth_model(device)
    pairwise_vp_init, pairwise_vs_init, pairwise_q_init = perturb_initial(
        base.vp, base.vs, base.q, seed=args.seed + 1, q_scale=1.0
    )

    opt = torch.optim.Adam(bridge.physics_head.parameters(), lr=args.lr)
    history: list[dict] = []

    best_vp_rmse = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            bridge, train_loader, opt, device,
            args.vp_sup_weight, args.waveform_weight,
            args.unrolled_weight, args.unrolled_steps, args.unrolled_step_size,
            args.pairwise_weight, args.pairwise_margin,
            pairwise_vp_init, pairwise_vs_init, pairwise_q_init,
        )
        va = run_epoch(
            bridge, val_loader, None, device,
            0.0, args.waveform_weight,
            args.unrolled_weight, args.unrolled_steps, args.unrolled_step_size,
            args.pairwise_weight, args.pairwise_margin,
            pairwise_vp_init, pairwise_vs_init, pairwise_q_init,
        )
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        vp_rmse = va.get("rmse_vp_rmse", va.get("rmse_vp", float("nan")))
        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_vp_rmse={vp_rmse:.4f}",
            flush=True,
        )
        if vp_rmse < best_vp_rmse:
            best_vp_rmse = vp_rmse
            torch.save(
                {
                    "physics_head": bridge.physics_head.state_dict(),
                    "args": vars(args),
                    "ckpt_args": ckpt_args,
                    "trainable_params": bridge.trainable_parameter_count(),
                },
                out_dir / "best_physics_head.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    report = {
        "checkpoint": str(ckpt),
        "trainable_params": bridge.trainable_parameter_count(),
        "total_params": bridge.total_parameter_count(),
        "best_val_vp_rmse": best_vp_rmse,
        "final_val": history[-1]["val"] if history else {},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[zhizi-inv] -> {out_dir}")


if __name__ == "__main__":
    main()
