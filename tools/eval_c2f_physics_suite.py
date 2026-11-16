#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-contribution suite for a c2f checkpoint (val set, det-gated, th 0.3, tol 0.5 s).

Reports: gated accuracy, head ablation (full / wave-only / refine-only),
Huygens kernel parameters per layer, and ρ-peak onset hit rates.

  python tools/eval_c2f_physics_suite.py --checkpoint outputs/stead_c2f_s1_B/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.picking_metrics import (  # noqa: E402
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402
from tools.train_stead_picking import move_batch_to_device  # noqa: E402

VARIANTS = ("full", "wave_only", "refine_only")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tolerance-sec", type=float, default=0.5)
    return p.parse_args()


def variant_logits(model, out, name):
    if name == "full":
        return out["p"], out["s"]
    if name == "wave_only":
        if model.fusion == "sum":
            ws = torch.nn.functional.softplus(model.wave_logit_scale)
            return ws * out["p_wave"], ws * out["s_wave"]
        return out["p_wave"], out["s_wave"]
    return out["p_ref"], out["s_ref"]


def kernel_params(model):
    rows = {}
    for branch in ("shared", "p_layers", "s_layers"):
        rows[branch] = []
        for blk in getattr(model, branch):
            k = blk.kernel
            rows[branch].append({
                "c": float(k.effective_wave_speed().float().mean()),
                "gamma": float(k.effective_gamma().float().mean()),
                "omega": float(k.effective_omega().float().mean()),
            })
    cp = sum(r["c"] for r in rows["p_layers"]) / len(rows["p_layers"])
    cs = sum(r["c"] for r in rows["s_layers"]) / len(rows["s_layers"])
    rows["vp_over_vs"] = cp / max(cs, 1e-6)
    return rows


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _, _ = load_model(args.checkpoint, device)
    ds = STEADPickingDataset("val", seq_len=6000, max_event_traces=4000, max_noise_traces=4000,
                             seed=42, load_geometry=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    th, tb = args.pick_threshold, max(1, int(round(args.tolerance_sec * 100)))
    accs = {v: EvalAccumulator() for v in VARIANTS}
    coarse_bin = 60.0 / model.coarse_len
    hit = {"p_0.5": 0, "s_0.5": 0, "p_0.2": 0, "s_0.2": 0, "n_p": 0, "n_s": 0}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det_prob = torch.sigmoid(out["det"])
        det_pred = det_prob >= args.det_threshold
        gate = det_pred.float().unsqueeze(-1)
        det_true = (batch["det"] > 0.5).cpu()
        yv = {k: batch[k].cpu() for k in ("p_valid", "s_valid", "p_idx", "s_idx")}
        for v in VARIANTS:
            pl, sl = variant_logits(model, out, v)
            p, s = apply_p_before_s_constraint(torch.sigmoid(pl.float()) * gate,
                                               torch.sigmoid(sl.float()) * gate, th)
            p, s = p.cpu(), s.cpu()
            acc = accs[v]
            update_detection_counts(acc, det_pred.cpu(), det_true)
            update_picking_counts(acc.p, p, det_pred.cpu(), det_true, yv["p_valid"] > 0, yv["p_idx"], th, tb, 6000)
            update_picking_counts(acc.s, s, det_pred.cpu(), det_true, yv["s_valid"] > 0, yv["s_idx"], th, tb, 6000)

        rho = out["rho"].float()
        top2 = rho.topk(2, dim=-1).indices.float() * coarse_bin
        for ph in ("p", "s"):
            valid = (batch[f"{ph}_valid"] > 0) & (batch["det"] > 0.5)
            t_true = batch[f"{ph}_idx"].float() / 100.0
            d = (top2 - t_true.unsqueeze(-1)).abs().amin(-1)
            hit[f"{ph}_0.5"] += int(((d <= 0.5) & valid).sum())
            hit[f"{ph}_0.2"] += int(((d <= 0.2) & valid).sum())
            hit[f"n_{ph}"] += int(valid.sum())

    heads = {v: finalize_metrics(accs[v]) for v in VARIANTS}
    rho_hit = {k: hit[k] / max(hit[f"n_{k[0]}"], 1) for k in ("p_0.5", "s_0.5", "p_0.2", "s_0.2")}
    report = {
        "checkpoint": str(args.checkpoint),
        "fusion": model.fusion,
        "heads": {v: {k: heads[v][k] for k in ("det_f1", "p_f1", "s_f1")} for v in VARIANTS},
        "rho_top2_hit": rho_hit,
        "kernel": kernel_params(model),
    }
    for v in VARIANTS:
        m = heads[v]
        print(f"  {v:12s} P={m['p_f1']:.4f} S={m['s_f1']:.4f}", flush=True)
    print(f"  rho hit {rho_hit}  vp/vs={report['kernel']['vp_over_vs']:.2f}", flush=True)
    out_path = args.output or args.checkpoint.parent / "physics_suite.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[done] → {out_path}", flush=True)


if __name__ == "__main__":
    main()
