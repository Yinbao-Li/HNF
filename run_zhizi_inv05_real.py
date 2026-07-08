#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inv05-real: STEAD test waveforms + run20 picks -> Zhizi macro/residual init -> TT refine.

Compares on the same observed picks:
  1) perturb init -> L-BFGS/Adam travel-time refine
  2) Zhizi init  -> same refine
Metrics: pick MAE vs catalog; travel-time misfit before/after refine; whether Zhizi wins refine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.inversion_1d import default_synth_model, travel_time_phase
from hnf.inversion_baselines import invert_hnf_adam, invert_lbfgs_torch
from hnf.inv_plot import perturb_initial
from hnf.picking_metrics import idx_to_sec
from hnf.picking_prior import run_picking_on_batch
from hnf.stead_picking_dataset import STEADPickingDataset
from hnf.zhizi_inversion_bridge import ZhiziInversionBridge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inv05-real: STEAD + Zhizi inversion bridge")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--physics-head", default="outputs/zhizi_inversion_bridge_macro/best_physics_head.pt")
    p.add_argument("--head-mode", choices=["residual", "macro"], default="macro")
    p.add_argument("--output-dir", default="outputs/zhizi_inv05_real_macro")
    p.add_argument("--max-events", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=None, help="Default: from checkpoint")
    p.add_argument(
        "--infer-seq-len",
        type=int,
        default=None,
        help="Bridge feature downsample length (default 600); picks use full seq_len",
    )
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--nominal-distance-km", type=float, default=50.0)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument(
        "--obs-fallback",
        action="store_true",
        help="Use catalog picks when model pick error > threshold",
    )
    p.add_argument("--pick-fallback-sec", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def time_misfit(earth, source_depth, distances, obs_tp, obs_ts):
    src = torch.tensor(source_depth, dtype=earth.vp.dtype, device=earth.vp.device)
    tp = travel_time_phase(earth, "P", src, distances)
    ts = travel_time_phase(earth, "S", src, distances)
    return float(torch.mean((tp - obs_tp) ** 2) + torch.mean((ts - obs_ts) ** 2))


def refine_tt(depths, vp0, vs0, q0, source_depth, distances, obs_tp, obs_ts):
    res = invert_lbfgs_torch(
        depths, vp0, vs0, q0, source_depth, distances, obs_tp, obs_ts, max_iter=80
    )
    if not torch.isfinite(res.earth.vp).all():
        res = invert_hnf_adam(
            depths, vp0, vs0, q0, source_depth, distances,
            {"tp": obs_tp, "ts": obs_ts}, steps=400,
        )
    return res


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
    ckpt_seq_len = args.seq_len or int(ckpt_args.get("seq_len", 600))
    bridge_infer_len = args.infer_seq_len if args.infer_seq_len is not None else min(ckpt_seq_len, 600)
    embed_dim = int(ckpt_args.get("embed_dim", 64))
    base = default_synth_model(device)
    n_layers = base.n_layers
    bridge = ZhiziInversionBridge(
        backbone=backbone,
        n_layers=n_layers,
        embed_dim=embed_dim,
        hidden=48,
        freeze_backbone=True,
        infer_seq_len=bridge_infer_len,
        head_mode=args.head_mode,
    ).to(device)
    state = torch.load(head_ckpt, map_location=device, weights_only=False)
    bridge.physics_head.load_state_dict(state["physics_head"])
    bridge.eval()

    ds = STEADPickingDataset("test", seq_len=ckpt_seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    distances = torch.tensor([args.nominal_distance_km], dtype=torch.float32, device=device)
    vp_pert, vs_pert, q_init = perturb_initial(
        base.vp, base.vs, base.q, seed=42, q_scale=1.0
    )

    pick_p_err: list[float] = []
    pick_s_err: list[float] = []
    rows: list[dict] = []
    n_seen = 0

    for batch in loader:
        if n_seen >= args.max_events:
            break
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        is_event = float(batch["det"][0]) > 0.5
        if not is_event:
            continue
        if batch["p_valid"][0] <= 0 or batch["s_valid"][0] <= 0:
            continue

        n_seen += 1
        trace_len = x.shape[1]
        gt_p = idx_to_sec(int(batch["p_idx"][0]), trace_len)
        gt_s = idx_to_sec(int(batch["s_idx"][0]), trace_len)

        picks = run_picking_on_batch(
            backbone, x, t,
            pick_threshold=args.pick_threshold,
            det_threshold=args.det_threshold,
            infer_seq_len=None,
        )
        tp = picks["tp_sec"][0]
        ts = picks["ts_sec"][0]
        det_ok = float(picks["det_prob"][0]) >= args.det_threshold
        raw_tp, raw_ts = tp, ts

        if raw_tp is not None:
            pick_p_err.append(abs(raw_tp - gt_p))
        if raw_ts is not None:
            pick_s_err.append(abs(raw_ts - gt_s))

        if args.obs_fallback:
            if tp is None or (tp is not None and abs(tp - gt_p) > args.pick_fallback_sec):
                tp = gt_p
            if ts is None or (ts is not None and abs(ts - gt_s) > args.pick_fallback_sec):
                ts = gt_s

        row: dict = {
            "idx": n_seen - 1,
            "det_prob": float(picks["det_prob"][0]),
            "det_ok": det_ok,
            "catalog_tp_sec": gt_p,
            "catalog_ts_sec": gt_s,
            "model_tp_sec": raw_tp,
            "model_ts_sec": raw_ts,
            "pick_err_p": abs(raw_tp - gt_p) if raw_tp is not None else None,
            "pick_err_s": abs(raw_ts - gt_s) if raw_ts is not None else None,
            "kernel_vp": picks["kernel_vp"],
            "kernel_vs": picks["kernel_vs"],
        }

        if tp is None or ts is None or not det_ok:
            row["status"] = "skip_inversion"
            rows.append(row)
            continue

        obs_tp = torch.tensor([tp], device=device)
        obs_ts = torch.tensor([ts], device=device)

        with torch.no_grad():
            out, _ = bridge.forward_event(x, t, include_picks=True)
        zhizi_earth = bridge.physics_head.earth(out, base.depths, base.q)
        zhizi_init_tt = time_misfit(zhizi_earth, args.source_depth, distances, obs_tp, obs_ts)
        pert_init_tt = time_misfit(
            type(zhizi_earth)(depths=base.depths, vp=vp_pert, vs=vs_pert, q=q_init),
            args.source_depth, distances, obs_tp, obs_ts,
        )

        zh_ref = refine_tt(
            base.depths, zhizi_earth.vp.detach(), zhizi_earth.vs.detach(), q_init,
            args.source_depth, distances, obs_tp, obs_ts,
        )
        pe_ref = refine_tt(
            base.depths, vp_pert, vs_pert, q_init,
            args.source_depth, distances, obs_tp, obs_ts,
        )

        row.update({
            "status": "ok",
            "zhizi_init_vp": zhizi_earth.vp.detach().cpu().tolist(),
            "zhizi_refined_vp": zh_ref.earth.vp.cpu().tolist(),
            "perturb_refined_vp": pe_ref.earth.vp.cpu().tolist(),
            "zhizi_init_tt": zhizi_init_tt,
            "perturb_init_tt": pert_init_tt,
            "zhizi_refined_tt": zh_ref.time_misfit,
            "perturb_refined_tt": pe_ref.time_misfit,
            "zhizi_refine_wins": float(zh_ref.time_misfit) < float(pe_ref.time_misfit),
            "obs_tp_sec": tp,
            "obs_ts_sec": ts,
        })
        rows.append(row)
        print(
            f"[{n_seen}/{args.max_events}] pick_p={row['pick_err_p'] or float('nan'):.3f}s "
            f"init z_tt={zhizi_init_tt:.4f} p_tt={pert_init_tt:.4f} | "
            f"ref z_tt={zh_ref.time_misfit:.4f} p_tt={pe_ref.time_misfit:.4f}",
            flush=True,
        )

    ok_rows = [r for r in rows if r.get("status") == "ok"]

    def finite_mean(key: str) -> float:
        vals = [r[key] for r in ok_rows if key in r and r[key] == r[key]]
        return sum(vals) / max(len(vals), 1)

    summary = {
        "n_events": len(rows),
        "n_inverted": len(ok_rows),
        "checkpoint": str(ckpt),
        "physics_head": str(head_ckpt),
        "head_mode": args.head_mode,
        "seq_len": ckpt_seq_len,
        "infer_seq_len": bridge_infer_len,
        "obs_fallback": args.obs_fallback,
        "nominal_distance_km": args.nominal_distance_km,
        "pick_mae_p": sum(pick_p_err) / max(len(pick_p_err), 1),
        "pick_mae_s": sum(pick_s_err) / max(len(pick_s_err), 1),
        "zhizi_init_tt_mean": finite_mean("zhizi_init_tt"),
        "perturb_init_tt_mean": finite_mean("perturb_init_tt"),
        "zhizi_refined_tt_mean": finite_mean("zhizi_refined_tt"),
        "perturb_refined_tt_mean": finite_mean("perturb_refined_tt"),
        "zhizi_refine_win_frac": sum(1 for r in ok_rows if r.get("zhizi_refine_wins")) / max(len(ok_rows), 1),
        "events": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    print(f"[inv05-real] -> {out_dir}")


if __name__ == "__main__":
    main()
