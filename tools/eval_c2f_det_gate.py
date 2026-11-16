#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detection-gated picking for c2f: choose gate on val, report test once.

  none : raw pick probabilities
  hard : zero picks when det_prob < det_threshold (EQTransformer predictor behaviour)
  soft : pick_prob * det_prob

  python tools/eval_c2f_det_gate.py --checkpoint outputs/stead_c2f_750_v4full/best.pt
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

GATES = ("none", "hard", "soft")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_750_v4full/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paper_stead_final_compare"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--max-val", type=int, default=8000)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tolerances", default="0.1,0.2,0.3,0.5")
    p.add_argument("--force-gate", default=None, choices=["none", "hard", "soft"],
                   help="use this gate on test instead of the val-selected one")
    p.add_argument("--output-name", default="det_gate.json")
    return p.parse_args()


def gate(probs, det_prob, kind, det_th):
    if kind == "hard":
        return probs * (det_prob >= det_th).float().unsqueeze(-1)
    if kind == "soft":
        return probs * det_prob.unsqueeze(-1)
    return probs


@torch.no_grad()
def evaluate(model, ds, device, args, gates, tols):
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    accs = {(g, t): EvalAccumulator() for g in gates for t in tols}
    th = args.pick_threshold
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det_prob = torch.sigmoid(out["det"])
        det_pred = (det_prob >= args.det_threshold).cpu()
        det_true = (batch["det"] > 0.5).cpu()
        p_raw, s_raw = torch.sigmoid(out["p"]), torch.sigmoid(out["s"])
        yv = {k: batch[k].cpu() for k in ("p_valid", "s_valid", "p_idx", "s_idx")}
        for g in gates:
            p, s = apply_p_before_s_constraint(
                gate(p_raw, det_prob, g, args.det_threshold), gate(s_raw, det_prob, g, args.det_threshold), th
            )
            p, s = p.float().cpu(), s.float().cpu()
            for t in tols:
                acc = accs[(g, t)]
                tb = max(1, int(round(t * 100.0)))
                update_detection_counts(acc, det_pred, det_true)
                update_picking_counts(acc.p, p, det_pred, det_true, yv["p_valid"] > 0, yv["p_idx"], th, tb, 6000)
                update_picking_counts(acc.s, s, det_pred, det_true, yv["s_valid"] > 0, yv["s_idx"], th, tb, 6000)
    return {g: {f"{t:.1f}": finalize_metrics(accs[(g, t)]) for t in tols} for g in gates}


def show(tag, res):
    for g, rows in res.items():
        for t, m in rows.items():
            print(f"  {tag} gate={g:4s} tol={t}s P={m['p_f1']:.4f} (prec {m['p_precision']:.4f} rec {m['p_recall']:.4f}) "
                  f"S={m['s_f1']:.4f} (prec {m['s_precision']:.4f} rec {m['s_recall']:.4f})", flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    tols = [float(x) for x in args.tolerances.split(",")]
    model, _, _ = load_model(args.checkpoint, device)

    val_ds = STEADPickingDataset("val", seq_len=6000, max_event_traces=args.max_val // 2,
                                 max_noise_traces=args.max_val // 2, seed=42, load_geometry=False)
    val = evaluate(model, val_ds, device, args, GATES, [0.5])
    show("val", val)
    score = {g: 0.5 * (val[g]["0.5"]["p_f1"] + val[g]["0.5"]["s_f1"]) for g in GATES}
    chosen = args.force_gate or max(score, key=score.get)
    print(f"[gate] val-selected gate = {chosen}  (mean P/S F1 @0.5s: "
          + ", ".join(f"{g}={v:.4f}" for g, v in score.items()) + ")", flush=True)

    test_ds = STEADPickingDataset("test", seq_len=6000, load_geometry=False)
    test = evaluate(model, test_ds, device, args, [chosen], tols)
    show("test", test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / args.output_name
    out.write_text(json.dumps({"val": val, "val_selected_gate": chosen, "test": test,
                               "pick_threshold": args.pick_threshold, "det_threshold": args.det_threshold},
                              indent=2))
    print(f"[done] → {out}", flush=True)


if __name__ == "__main__":
    main()
