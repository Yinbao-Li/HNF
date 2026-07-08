#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Route A validation: Zhizi vp/vs prior + damped Gauss-Newton refinement.

Compares per event (synthetic, ground-truth vp available):
  1) perturb init  -> GN
  2) Zhizi one-shot (no refine)
  3) Zhizi init    -> GN

Core question: does Zhizi velocity carry physical meaning as a GN initializer?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.inversion_1d import LayeredEarth1D, default_synth_model
from hnf.inv_plot import perturb_initial
from hnf.route_a_refine import RouteARow, build_verdict, refine_gn, rmse_vs_true
from hnf.zhizi_inversion_bridge import ZhiziInversionBridge
from hnf.zhizi_inversion_dataset import ZhiziInversionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route A: Zhizi init + GN refine validation")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/route_a_validate")
    p.add_argument("--n-test", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gn-iters", type=int, default=25)
    p.add_argument("--gn-iters-fast", type=int, default=5, help="Short GN for convergence check")
    p.add_argument("--infer-seq-len", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def mean_key(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(len(rows), 1)


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

    bridge = ZhiziInversionBridge(
        backbone=backbone,
        n_layers=n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=args.infer_seq_len,
    ).to(device)
    state = torch.load(head_ckpt, map_location=device, weights_only=False)
    bridge.physics_head.load_state_dict(state["physics_head"])
    bridge.eval()

    test_ds = ZhiziInversionDataset(
        n_samples=args.n_test, seq_len=args.infer_seq_len, seed=args.seed, device=device
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    rows_full: list[RouteARow] = []
    rows_fast: list[RouteARow] = []
    detail: list[dict] = []

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

        base = default_synth_model(device)
        vp_pert, vs_pert, q_init = perturb_initial(
            base.vp, base.vs, base.q, seed=args.seed + idx * 9973, q_scale=1.0
        )

        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True)
        zhizi_earth = bridge.physics_head.earth(out, depths, true_q)
        vp_zh = zhizi_earth.vp.detach().clone()
        vs_zh = zhizi_earth.vs.detach().clone()

        zhizi_init_rmse = rmse_vs_true(true_earth, zhizi_earth)
        pert_earth = LayeredEarth1D(depths=depths, vp=vp_pert, vs=vs_pert, q=q_init)
        pert_init_rmse = rmse_vs_true(true_earth, pert_earth)

        gn_zh = refine_gn(
            depths, vp_zh, vs_zh, q_init, source_depth, distances, obs_tp, obs_ts,
            n_iter=args.gn_iters,
        )
        gn_pert = refine_gn(
            depths, vp_pert, vs_pert, q_init, source_depth, distances, obs_tp, obs_ts,
            n_iter=args.gn_iters,
        )
        gn_zh_fast = refine_gn(
            depths, vp_zh, vs_zh, q_init, source_depth, distances, obs_tp, obs_ts,
            n_iter=args.gn_iters_fast,
        )
        gn_pert_fast = refine_gn(
            depths, vp_pert, vs_pert, q_init, source_depth, distances, obs_tp, obs_ts,
            n_iter=args.gn_iters_fast,
        )

        row = RouteARow(
            idx=idx,
            zhizi_init_vp_rmse=zhizi_init_rmse["vp_rmse"],
            perturb_init_vp_rmse=pert_init_rmse["vp_rmse"],
            zhizi_refined_vp_rmse=rmse_vs_true(true_earth, gn_zh.earth)["vp_rmse"],
            perturb_refined_vp_rmse=rmse_vs_true(true_earth, gn_pert.earth)["vp_rmse"],
            zhizi_init_tt=gn_zh.history[0]["loss"] if gn_zh.history else float("nan"),
            perturb_init_tt=gn_pert.history[0]["loss"] if gn_pert.history else float("nan"),
            zhizi_refined_tt=gn_zh.time_misfit,
            perturb_refined_tt=gn_pert.time_misfit,
            zhizi_refine_sec=gn_zh.wall_sec,
            perturb_refine_sec=gn_pert.wall_sec,
            gn_iters=args.gn_iters,
        )
        rows_full.append(row)

        rows_fast.append(RouteARow(
            idx=idx,
            zhizi_init_vp_rmse=zhizi_init_rmse["vp_rmse"],
            perturb_init_vp_rmse=pert_init_rmse["vp_rmse"],
            zhizi_refined_vp_rmse=rmse_vs_true(true_earth, gn_zh_fast.earth)["vp_rmse"],
            perturb_refined_vp_rmse=rmse_vs_true(true_earth, gn_pert_fast.earth)["vp_rmse"],
            zhizi_init_tt=gn_zh_fast.history[0]["loss"] if gn_zh_fast.history else float("nan"),
            perturb_init_tt=gn_pert_fast.history[0]["loss"] if gn_pert_fast.history else float("nan"),
            zhizi_refined_tt=gn_zh_fast.time_misfit,
            perturb_refined_tt=gn_pert_fast.time_misfit,
            zhizi_refine_sec=gn_zh_fast.wall_sec,
            perturb_refine_sec=gn_pert_fast.wall_sec,
            gn_iters=args.gn_iters_fast,
        ))

        detail.append({
            "idx": idx,
            "true_vp": true_vp.cpu().tolist(),
            "zhizi_init_vp": vp_zh.cpu().tolist(),
            "perturb_init_vp": vp_pert.cpu().tolist(),
            "zhizi_refined_vp": gn_zh.earth.vp.cpu().tolist(),
            "perturb_refined_vp": gn_pert.earth.vp.cpu().tolist(),
            "zhizi_init_vp_rmse": row.zhizi_init_vp_rmse,
            "perturb_init_vp_rmse": row.perturb_init_vp_rmse,
            "zhizi_refined_vp_rmse": row.zhizi_refined_vp_rmse,
            "perturb_refined_vp_rmse": row.perturb_refined_vp_rmse,
        })
        print(
            f"[{idx+1}/{args.n_test}] init: zhizi={row.zhizi_init_vp_rmse:.3f} "
            f"pert={row.perturb_init_vp_rmse:.3f} | "
            f"refined: zhizi={row.zhizi_refined_vp_rmse:.3f} pert={row.perturb_refined_vp_rmse:.3f}",
            flush=True,
        )

    verdict = build_verdict(rows_full, rows_fast)
    summary = {
        "route": "A",
        "question": "智子学到的速度是否具有物理意义（作为 GN 初值）？",
        "n_test": len(rows_full),
        "gn_iters_full": args.gn_iters,
        "gn_iters_fast": args.gn_iters_fast,
        "checkpoint": str(ckpt),
        "physics_head": str(head_ckpt),
        "init": {
            "zhizi_vp_rmse_mean": mean_key([asdict(r) for r in rows_full], "zhizi_init_vp_rmse"),
            "perturb_vp_rmse_mean": mean_key([asdict(r) for r in rows_full], "perturb_init_vp_rmse"),
            "zhizi_better_frac": verdict.init_zhizi_better_frac,
            "rmse_ratio_zhizi_over_perturb": verdict.init_rmse_ratio,
        },
        "refined_full_gn": {
            "zhizi_vp_rmse_mean": mean_key([asdict(r) for r in rows_full], "zhizi_refined_vp_rmse"),
            "perturb_vp_rmse_mean": mean_key([asdict(r) for r in rows_full], "perturb_refined_vp_rmse"),
            "zhizi_better_frac": verdict.refined_zhizi_better_frac,
            "rmse_ratio_zhizi_over_perturb": verdict.refined_rmse_ratio,
            "zhizi_tt_misfit_mean": mean_key([asdict(r) for r in rows_full], "zhizi_refined_tt"),
            "perturb_tt_misfit_mean": mean_key([asdict(r) for r in rows_full], "perturb_refined_tt"),
        },
        "refined_fast_gn": {
            "zhizi_vp_rmse_mean": mean_key([asdict(r) for r in rows_fast], "zhizi_refined_vp_rmse"),
            "perturb_vp_rmse_mean": mean_key([asdict(r) for r in rows_fast], "perturb_refined_vp_rmse"),
            "rmse_ratio_zhizi_over_perturb": verdict.convergence_at_5_ratio,
        },
        "verdict": asdict(verdict),
        "per_event": [asdict(r) for r in rows_full],
        "detail": detail,
    }
    (out_dir / "route_a_report.json").write_text(json.dumps(summary, indent=2))

    # convergence plot
    fig, ax = plt.subplots(figsize=(7, 4))
    init_z = [r.zhizi_init_vp_rmse for r in rows_full]
    init_p = [r.perturb_init_vp_rmse for r in rows_full]
    ref_z = [r.zhizi_refined_vp_rmse for r in rows_full]
    ref_p = [r.perturb_refined_vp_rmse for r in rows_full]
    fast_z = [r.zhizi_refined_vp_rmse for r in rows_fast]
    fast_p = [r.perturb_refined_vp_rmse for r in rows_fast]
    xpos = range(len(rows_full))
    ax.scatter(xpos, init_z, s=18, alpha=0.6, label="Zhizi init", c="C0")
    ax.scatter(xpos, init_p, s=18, alpha=0.6, label="Perturb init", c="C1")
    ax.scatter(xpos, fast_z, s=28, marker="^", label=f"Zhizi+GN({args.gn_iters_fast})", c="C0")
    ax.scatter(xpos, fast_p, s=28, marker="v", label=f"Perturb+GN({args.gn_iters_fast})", c="C1")
    ax.scatter(xpos, ref_z, s=36, marker="*", label=f"Zhizi+GN({args.gn_iters})", c="C0", edgecolors="k", linewidths=0.3)
    ax.scatter(xpos, ref_p, s=36, marker="D", label=f"Perturb+GN({args.gn_iters})", c="C1", edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Event index")
    ax.set_ylabel("Vp RMSE (km/s)")
    ax.set_title("Route A: Zhizi velocity as GN initializer")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "route_a_vp_rmse.png", dpi=140)
    plt.close(fig)

    print("\n=== Route A Verdict ===")
    print(f"物理意义: {'是' if verdict.physically_meaningful else '否（或尚不充分）'}")
    print(f"理由: {verdict.rationale}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_event", "detail")}, indent=2))
    print(f"[route-a] -> {out_dir}")


if __name__ == "__main__":
    main()
