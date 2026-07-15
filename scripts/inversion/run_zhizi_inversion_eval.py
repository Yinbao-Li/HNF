#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Physics Decoder vs classical travel-time baselines.

Uses synthetic multi-station waveforms (same generator as training) and
reports Vp/Vs RMSE + travel-time misfit per method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.analyze_stead_picking import load_model
from hnf.inversion_1d import LayeredEarth1D, default_synth_model, model_rmse, travel_time_phase
from hnf.inversion_baselines import invert_lbfgs_torch
from hnf.inv_plot import perturb_initial
from hnf.physics_decoder import PhysicsDecoder
from hnf.zhizi_inversion_dataset import ZhiziInversionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Physics Decoder")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/zhizi_inversion_bridge")
    p.add_argument("--n-test", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def time_misfit(earth, source_depth, distances, obs_tp, obs_ts):
    src = torch.tensor(source_depth, dtype=earth.vp.dtype, device=earth.vp.device)
    tp = travel_time_phase(earth, "P", src, distances)
    ts = travel_time_phase(earth, "S", src, distances)
    return float(torch.mean((tp - obs_tp) ** 2) + torch.mean((ts - obs_ts) ** 2))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint)
    head_ckpt = Path(args.physics_head)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    if not head_ckpt.exists():
        raise FileNotFoundError(head_ckpt)

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
    ).to(device)
    state = torch.load(head_ckpt, map_location=device)
    bridge.physics_head.load_state_dict(state["physics_head"])
    bridge.eval()

    test_ds = ZhiziInversionDataset(
        n_samples=args.n_test, seq_len=args.infer_seq_len, seed=args.seed, device=device
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    base = default_synth_model(device)
    vp_init, vs_init, q_init = perturb_initial(
        base.vp, base.vs, base.q, seed=args.seed + 1, q_scale=1.0
    )

    rows: list[dict] = []
    for idx, batch in enumerate(loader):
        x = batch["x"][0].to(device)
        t = batch["t"].to(device)
        obs_tp = batch["obs_tp"][0].to(device)
        obs_ts = batch["obs_ts"][0].to(device)
        true_vp = batch["true_vp"][0].to(device)
        true_vs = batch["true_vs"][0].to(device)
        true_q = batch["true_q"][0].to(device)
        depths = batch["depths"][0].to(device)
        distances = batch["distances"][0].to(device)
        source_depth = float(batch["source_depth"][0])
        true_earth = LayeredEarth1D(depths=depths, vp=true_vp, vs=true_vs, q=true_q)

        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True)
        zhizi_earth = bridge.physics_head.earth(out, depths, true_q)
        zhizi_rmse = model_rmse(true_earth, zhizi_earth)
        zhizi_tt = time_misfit(zhizi_earth, source_depth, distances, obs_tp, obs_ts)
        zhizi_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        gn_res = invert_lbfgs_torch(
            depths, vp_init, vs_init, q_init,
            source_depth, distances, obs_tp, obs_ts,
            max_iter=80,
        )
        gn_earth = gn_res.earth
        gn_rmse = model_rmse(true_earth, gn_earth)
        gn_tt = time_misfit(gn_earth, source_depth, distances, obs_tp, obs_ts)
        gn_sec = time.perf_counter() - t0

        rows.append({
            "idx": idx,
            "zhizi_vp_rmse": zhizi_rmse["vp_rmse"],
            "zhizi_vs_rmse": zhizi_rmse["vs_rmse"],
            "zhizi_tt_misfit": zhizi_tt,
            "zhizi_wall_sec": zhizi_sec,
            "gn_vp_rmse": gn_rmse["vp_rmse"],
            "gn_vs_rmse": gn_rmse["vs_rmse"],
            "gn_tt_misfit": gn_tt,
            "gn_wall_sec": gn_sec,
        })
        print(
            f"[{idx+1}/{args.n_test}] zhizi vp={zhizi_rmse['vp_rmse']:.3f} "
            f"gn vp={gn_rmse['vp_rmse']:.3f}",
            flush=True,
        )

    def mean_key(k: str) -> float:
        return sum(r[k] for r in rows) / max(len(rows), 1)

    summary = {
        "n_test": len(rows),
        "checkpoint": str(ckpt),
        "physics_head": str(head_ckpt),
        "trainable_params": state.get("trainable_params"),
        "zhizi": {
            "vp_rmse_mean": mean_key("zhizi_vp_rmse"),
            "vs_rmse_mean": mean_key("zhizi_vs_rmse"),
            "tt_misfit_mean": mean_key("zhizi_tt_misfit"),
            "wall_sec_mean": mean_key("zhizi_wall_sec"),
        },
        "gn_lbfgs": {
            "vp_rmse_mean": mean_key("gn_vp_rmse"),
            "vs_rmse_mean": mean_key("gn_vs_rmse"),
            "tt_misfit_mean": mean_key("gn_tt_misfit"),
            "wall_sec_mean": mean_key("gn_wall_sec"),
        },
        "per_event": rows,
    }
    (out_dir / "eval_compare.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_event"}, indent=2))
    print(f"[zhizi-eval] -> {out_dir / 'eval_compare.json'}")


if __name__ == "__main__":
    main()
