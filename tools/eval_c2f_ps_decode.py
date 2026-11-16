#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-ordered S decoding for c2f (det-gated): S searched only after P pick + gap.

Gap chosen on val (mean P/S F1 @ tol 0.5 s), test evaluated once at all tolerances.
Optional --s-window-sec truncates the S-branch Huygens window at inference (memory ablation).

  python tools/eval_c2f_ps_decode.py --checkpoint outputs/stead_c2f_750_v4full/best.pt
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_750_v4full/best.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/paper_stead_final_compare/ps_decode.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--gaps", default="none,0,0.25,0.5,1.0", help="'none' = original P-before-S rule")
    p.add_argument("--tolerances", default="0.1,0.2,0.3,0.5")
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--s-window-sec", type=float, default=None)
    p.add_argument("--p-window-sec", type=float, default=None)
    p.add_argument("--skip-test", action="store_true")
    return p.parse_args()


def ps_decode(p, s, th, gap_bins):
    """Mask S before (P pick + gap) whenever a P pick exists; returns (p, s)."""
    if gap_bins is None:
        return apply_p_before_s_constraint(p, s, th)
    p_max, p_idx = p.max(-1)
    t = torch.arange(s.size(-1), device=s.device).unsqueeze(0)
    allow = (t > (p_idx + gap_bins).unsqueeze(-1)) | (p_max < th).unsqueeze(-1)
    return p, s * allow.float()


@torch.no_grad()
def evaluate(model, ds, device, args, gaps, tols):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    accs = {(g, t): EvalAccumulator() for g in gaps for t in tols}
    th = args.pick_threshold
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det_prob = torch.sigmoid(out["det"])
        gate = (det_prob >= args.det_threshold).float().unsqueeze(-1)
        p_raw, s_raw = torch.sigmoid(out["p"]) * gate, torch.sigmoid(out["s"]) * gate
        det_pred = (det_prob >= args.det_threshold).cpu()
        det_true = (batch["det"] > 0.5).cpu()
        yv = {k: batch[k].cpu() for k in ("p_valid", "s_valid", "p_idx", "s_idx")}
        for g in gaps:
            gb = None if g == "none" else int(round(float(g) * 100))
            p, s = ps_decode(p_raw, s_raw, th, gb)
            p, s = p.float().cpu(), s.float().cpu()
            for t in tols:
                acc = accs[(g, t)]
                tb = max(1, int(round(t * 100)))
                update_detection_counts(acc, det_pred, det_true)
                update_picking_counts(acc.p, p, det_pred, det_true, yv["p_valid"] > 0, yv["p_idx"], th, tb, 6000)
                update_picking_counts(acc.s, s, det_pred, det_true, yv["s_valid"] > 0, yv["s_idx"], th, tb, 6000)
    return {g: {f"{t:.1f}": finalize_metrics(accs[(g, t)]) for t in tols} for g in gaps}


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, _, _ = load_model(args.checkpoint, device)
    if args.s_window_sec is not None:
        for blk in model.s_layers:
            blk.kernel.local_window_sec = float(args.s_window_sec)
    if args.p_window_sec is not None:
        for blk in model.p_layers:
            blk.kernel.local_window_sec = float(args.p_window_sec)
    gaps = [g.strip() for g in args.gaps.split(",")]
    tols = [float(x) for x in args.tolerances.split(",")]

    val_ds = STEADPickingDataset("val", seq_len=6000, max_event_traces=4000, max_noise_traces=4000,
                                 seed=42, load_geometry=False)
    val = evaluate(model, val_ds, device, args, gaps, sorted(set(tols) | {0.5}))
    for g in gaps:
        for t, m in val[g].items():
            print(f"  val gap={g} tol={t}s P={m['p_f1']:.4f} S={m['s_f1']:.4f} "
                  f"P_MAE={m['p_mae_sec']:.4f} S_MAE={m['s_mae_sec']:.4f}", flush=True)
    for g in gaps:
        m = val[g]["0.5"]
        print(f"  val gap={g:5s} P={m['p_f1']:.4f} S={m['s_f1']:.4f} (S prec {m['s_precision']:.4f} rec {m['s_recall']:.4f})",
              flush=True)
    chosen = max(gaps, key=lambda g: val[g]["0.5"]["p_f1"] + val[g]["0.5"]["s_f1"])
    print(f"[decode] val-selected gap = {chosen}", flush=True)
    report = {"val": val, "selected_gap": chosen, "s_window_sec": args.s_window_sec,
              "p_window_sec": args.p_window_sec}
    if not args.skip_test:
        test_ds = STEADPickingDataset("test", seq_len=6000, load_geometry=False)
        test = evaluate(model, test_ds, device, args, [chosen], tols)[chosen]
        for t, m in test.items():
            print(f"  test gap={chosen} tol={t}s P={m['p_f1']:.4f} S={m['s_f1']:.4f} "
                  f"(S prec {m['s_precision']:.4f} rec {m['s_recall']:.4f})", flush=True)
        report["test"] = test
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output}", flush=True)


if __name__ == "__main__":
    main()
