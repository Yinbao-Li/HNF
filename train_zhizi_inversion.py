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
from hnf.stead_zhizi_inversion_dataset import SteadZhiziInversionDataset
from hnf.zhizi_inversion_bridge import ZhiziInversionBridge, load_physics_head_state
from hnf.zhizi_inversion_dataset import ZhiziInversionDataset
from hnf.zhizi_inversion_loss import zhizi_inversion_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Zhizi inversion Physics Head")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/zhizi_inversion_bridge")
    p.add_argument("--dataset", choices=["synthetic", "stead", "mixed"], default="synthetic")
    p.add_argument("--geo-condition", action="store_true", help="Condition macro head on distance/depth")
    p.add_argument("--resume-physics-head", default="", help="Optional prior physics head (strict=False)")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=24)
    p.add_argument("--stead-max-train", type=int, default=400)
    p.add_argument("--stead-max-val", type=int, default=80)
    p.add_argument("--mixed-synth-frac", type=float, default=0.5)
    p.add_argument("--train-geo-only", action="store_true", help="Only train trunk.0 (geo input layer)")
    p.add_argument("--anchor-weight", type=float, default=0.01)
    p.add_argument("--stead-waveform-weight", type=float, default=None)
    p.add_argument("--stead-unrolled-weight", type=float, default=None)
    p.add_argument("--synth-unrolled-weight", type=float, default=None)
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


def _event_from_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x_batch = batch["x"].to(device)
    t = batch["t"].to(device)
    if x_batch.dim() == 4:
        return x_batch[0], t
    if x_batch.dim() == 3:
        return x_batch[0].unsqueeze(0), t
    raise ValueError(f"unexpected waveform batch shape: {tuple(x_batch.shape)}")


def _obs_times_from_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obs_tp = batch["obs_tp"].to(device)
    obs_ts = batch["obs_ts"].to(device)
    if obs_tp.dim() > 1:
        return obs_tp[0], obs_ts[0]
    return obs_tp, obs_ts


def _normalize_wf(wf: torch.Tensor) -> torch.Tensor:
    scale = wf.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    return wf / scale


def _val_score(va: dict[str, float], dataset: str) -> float:
    if dataset == "mixed":
        synth = va.get("synth_rmse_vp_rmse", va.get("rmse_vp_rmse", float("inf")))
        stead = va.get("stead_loss_unrolled_waveform", va.get("loss_unrolled_waveform", float("inf")))
        return 0.55 * synth + 0.45 * stead
    if dataset == "stead":
        return va.get("loss_unrolled_waveform", va.get("loss_waveform", va.get("loss", float("inf"))))
    return va.get("rmse_vp_rmse", va.get("rmse_vp", va.get("loss", float("inf"))))


def _apply_train_freeze(bridge: ZhiziInversionBridge, train_geo_only: bool) -> None:
    if not train_geo_only:
        return
    for name, param in bridge.physics_head.named_parameters():
        param.requires_grad = name.startswith("trunk.0.")
    n = sum(p.numel() for p in bridge.physics_head.parameters() if p.requires_grad)
    print(f"[zhizi-inv] train_geo_only: {n} params unfrozen", flush=True)


def run_epoch(
    bridge: ZhiziInversionBridge,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    dataset_kind: str,
    vp_sup_weight: float,
    anchor_weight: float,
    waveform_weight: float,
    unrolled_weight: float,
    stead_waveform_weight: float | None,
    stead_unrolled_weight: float | None,
    synth_unrolled_weight: float | None,
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
    n_synth = 0
    n_stead = 0
    synth_sums: dict[str, float] = {}
    stead_sums: dict[str, float] = {}
    swf = stead_waveform_weight if stead_waveform_weight is not None else waveform_weight
    sur = stead_unrolled_weight if stead_unrolled_weight is not None else unrolled_weight
    synr = synth_unrolled_weight if synth_unrolled_weight is not None else unrolled_weight
    for batch in loader:
        x_event, t = _event_from_batch(batch, device)
        obs_tp, obs_ts = _obs_times_from_batch(batch, device)
        depths = batch["depths"][0].to(device)
        true_q = batch["true_q"][0].to(device)
        distances = batch["distances"][0].to(device)
        source_depth = float(batch["source_depth"][0])
        true_vp = batch["true_vp"][0].to(device) if "true_vp" in batch else None
        true_vs = batch["true_vs"][0].to(device) if "true_vs" in batch else None
        geo = batch.get("geo")
        if geo is not None:
            geo = geo[0].to(device)

        if train:
            optimizer.zero_grad()

        output, rho_layers = bridge.forward_event(
            x_event, t, include_picks=True, geo=geo
        )
        loss, metrics = zhizi_inversion_loss(
            output,
            depths=depths,
            q=true_q,
            source_depth=source_depth,
            receiver_distances=distances,
            obs_tp=obs_tp,
            obs_ts=obs_ts,
            rho_layers=rho_layers[0],
            true_vp=true_vp,
            true_vs=true_vs,
            vp_sup_weight=vp_sup_weight if train else 0.0,
            anchor_weight=anchor_weight,
        )

        dt = float(t[1, 0] - t[0, 0]) if t.shape[0] > 1 else 0.01
        engine = DirectWaveForward(device=device, nt=t.shape[0], dt=dt)
        use_real_obs = true_vp is None
        wf_w = swf if use_real_obs else 0.0
        unr_w = sur if use_real_obs else synr

        if wf_w > 0:
            pred_earth = bridge.physics_head.earth(output, depths, true_q)
            if use_real_obs:
                obs_wf = _normalize_wf(x_event[0, :, 2].unsqueeze(0))
                pred_wf = _normalize_wf(engine.simulate(pred_earth, source_depth, distances))
            else:
                true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=true_q)
                pred_wf = engine.simulate(pred_earth, source_depth, distances)
                obs_wf = engine.simulate(true_earth, source_depth, distances)
            loss_wf = torch.mean((pred_wf - obs_wf) ** 2)
            loss = loss + wf_w * loss_wf
            metrics["loss_waveform"] = float(loss_wf.detach())

        if unr_w > 0:
            if use_real_obs:
                obs_wf = _normalize_wf(x_event[0, :, 2].unsqueeze(0))
            else:
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
            if use_real_obs:
                loss_unrolled = refined_metrics["waveform"]
                metrics["loss_unrolled_waveform"] = float(loss_unrolled.detach())
            else:
                loss_unrolled = torch.mean((refined_earth.vp - true_vp) ** 2)
                metrics["loss_unrolled_vp"] = float(loss_unrolled.detach())
                metrics["loss_unrolled_waveform"] = float(refined_metrics["waveform"].detach())
            loss = loss + unr_w * loss_unrolled

        if pairwise_weight > 0 and true_vp is not None:
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

        if train and torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.physics_head.parameters(), 1.0)
            optimizer.step()

        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + v
        if use_real_obs:
            n_stead += 1
            for k, v in metrics.items():
                stead_sums[k] = stead_sums.get(k, 0.0) + v
        else:
            n_synth += 1
            for k, v in metrics.items():
                synth_sums[k] = synth_sums.get(k, 0.0) + v
        n += 1

    out = {k: v / max(n, 1) for k, v in sums.items()}
    if n_synth:
        for k, v in synth_sums.items():
            out[f"synth_{k}"] = v / n_synth
        if "synth_rmse_vp_rmse" not in out and "rmse_vp_rmse" in synth_sums:
            out["synth_rmse_vp_rmse"] = synth_sums["rmse_vp_rmse"] / n_synth
        elif "synth_rmse_vp_rmse" not in out and "synth_rmse_vp" in synth_sums:
            out["synth_rmse_vp_rmse"] = synth_sums["synth_rmse_vp"] / n_synth
    if n_stead:
        for k, v in stead_sums.items():
            out[f"stead_{k}"] = v / n_stead
        if "stead_loss_unrolled_waveform" not in out and "loss_unrolled_waveform" in stead_sums:
            out["stead_loss_unrolled_waveform"] = stead_sums["loss_unrolled_waveform"] / n_stead
    return out


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
        geo_condition=args.geo_condition,
    ).to(device)

    if args.resume_physics_head:
        resume_path = Path(args.resume_physics_head)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            missing, unexpected, loaded = load_physics_head_state(
                bridge.physics_head, state["physics_head"]
            )
            print(
                f"[zhizi-inv] resumed physics head from {resume_path} "
                f"loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

    _apply_train_freeze(bridge, args.train_geo_only)

    print(
        f"[zhizi-inv] dataset={args.dataset} geo={args.geo_condition} "
        f"trainable params={bridge.trainable_parameter_count()} "
        f"total={bridge.total_parameter_count()}",
        flush=True,
    )

    if args.dataset == "synthetic":
        train_ds = ZhiziInversionDataset(
            n_samples=args.n_train, seq_len=args.seq_len, seed=args.seed, device=device
        )
        val_ds = ZhiziInversionDataset(
            n_samples=args.n_val, seq_len=args.seq_len, seed=args.seed + 999, device=device
        )
    elif args.dataset == "stead":
        train_ds = SteadZhiziInversionDataset(
            split="train",
            seq_len=args.seq_len,
            max_traces=args.stead_max_train,
            seed=args.seed,
            augment=True,
        )
        val_ds = SteadZhiziInversionDataset(
            split="val",
            seq_len=args.seq_len,
            max_traces=args.stead_max_val,
            seed=args.seed + 999,
            augment=False,
        )
    else:
        synth_train = ZhiziInversionDataset(
            n_samples=args.n_train, seq_len=args.seq_len, seed=args.seed, device=device
        )
        synth_val = ZhiziInversionDataset(
            n_samples=args.n_val, seq_len=args.seq_len, seed=args.seed + 999, device=device
        )
        stead_train = SteadZhiziInversionDataset(
            split="train",
            seq_len=args.seq_len,
            max_traces=args.stead_max_train,
            seed=args.seed,
            augment=True,
        )
        stead_val = SteadZhiziInversionDataset(
            split="val",
            seq_len=args.seq_len,
            max_traces=args.stead_max_val,
            seed=args.seed + 999,
            augment=False,
        )
        from torch.utils.data import ConcatDataset

        train_ds = ConcatDataset([synth_train, stead_train])
        val_ds = ConcatDataset([synth_val, stead_val])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    base = default_synth_model(device)
    pairwise_vp_init, pairwise_vs_init, pairwise_q_init = perturb_initial(
        base.vp, base.vs, base.q, seed=args.seed + 1, q_scale=1.0
    )

    opt = torch.optim.Adam(
        [p for p in bridge.physics_head.parameters() if p.requires_grad], lr=args.lr
    )
    history: list[dict] = []

    best_score = float("inf")
    epoch_args = (
        args.vp_sup_weight, args.anchor_weight, args.waveform_weight, args.unrolled_weight,
        args.stead_waveform_weight, args.stead_unrolled_weight, args.synth_unrolled_weight,
        args.unrolled_steps, args.unrolled_step_size,
        args.pairwise_weight, args.pairwise_margin,
        pairwise_vp_init, pairwise_vs_init, pairwise_q_init,
    )
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(bridge, train_loader, opt, device, args.dataset, *epoch_args)
        va_args = (0.0,) + epoch_args[1:]
        va = run_epoch(bridge, val_loader, None, device, args.dataset, *va_args)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        score = _val_score(va, args.dataset)
        vp_rmse = va.get("synth_rmse_vp_rmse", va.get("rmse_vp_rmse", float("nan")))
        stead_u = va.get("stead_loss_unrolled_waveform", va.get("loss_unrolled_waveform", float("nan")))
        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
            f"val_score={score:.4f} synth_vp_rmse={vp_rmse:.4f} stead_unrolled={stead_u:.4f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "physics_head": bridge.physics_head.state_dict(),
                    "args": vars(args),
                    "ckpt_args": ckpt_args,
                    "trainable_params": bridge.trainable_parameter_count(),
                    "geo_condition": args.geo_condition,
                    "train_geo_only": args.train_geo_only,
                },
                out_dir / "best_physics_head.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    report = {
        "checkpoint": str(ckpt),
        "dataset": args.dataset,
        "geo_condition": args.geo_condition,
        "trainable_params": bridge.trainable_parameter_count(),
        "total_params": bridge.total_parameter_count(),
        "best_val_score": best_score,
        "best_val_vp_rmse": history[-1]["val"].get("rmse_vp_rmse") if history else None,
        "final_val": history[-1]["val"] if history else {},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[zhizi-inv] -> {out_dir}")


if __name__ == "__main__":
    main()
