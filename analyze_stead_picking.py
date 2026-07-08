#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-pass STEAD test error analysis + threshold sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hnf.picking_metrics import (
    apply_p_before_s_constraint,
    det_pred_from_logits,
    idx_to_sec,
    tolerance_bins,
)
from hnf.picking_model import build_picking_model, load_picking_model_state
from hnf.stead_picking_dataset import STEADPickingDataset


def load_model(checkpoint: Path, device: torch.device, bypass_noise_cancel: bool = False):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = build_picking_model(
        embed_dim=args.get("embed_dim", 64),
        num_shared_layers=args.get("num_shared_layers", 2),
        num_branch_layers=args.get("num_branch_layers", 2),
        vp=args.get("vp", 8.0),
        vs=args.get("vs", 4.5),
        local_window_sec=args.get("local_window_sec", 15.0),
        dropout=args.get("dropout", 0.1),
        per_time_det=args.get("per_time_det", False),
        pick_head_hidden=args.get("pick_head_hidden", 24),
        pick_head_kernel=args.get("pick_head_kernel", 7),
        pick_head_layers=args.get("pick_head_layers", 3),
        multi_scale=args.get("multi_scale", False),
        num_anchors=int(args.get("num_anchors", 0)),
        residual_pick_head=args.get("residual_pick_head", True),
        residual_det_head=args.get("residual_det_head", False),
        enhanced_det_head=args.get("enhanced_det_head", False),
        noise_cancel=args.get("noise_cancel", False),
        noise_source_dim=args.get("noise_source_dim", 16),
        noise_det_pick_split=args.get("noise_det_pick_split", False),
        noise_pick_cues=args.get("noise_pick_cues", False),
    ).to(device)
    load_picking_model_state(model, ckpt["state_dict"], strict=False)
    model.bypass_noise_cancel = bypass_noise_cancel
    model.eval()
    return model, args


def pick_prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def analyze(
    checkpoint: Path,
    seq_len: int,
    device: torch.device,
    pick_thresholds: list[float],
    det_thresholds: list[float],
    bypass_noise_cancel: bool = False,
) -> dict:
    model, ckpt_args = load_model(checkpoint, device, bypass_noise_cancel=bypass_noise_cancel)
    ds = STEADPickingDataset("test", seq_len=seq_len)
    loader = DataLoader(ds, batch_size=int(ckpt_args.get("batch_size", 24)), shuffle=False)
    tol = tolerance_bins(seq_len, 0.5)

    det_acc = {d: Counter() for d in det_thresholds}
    pick_acc = {
        d: {p: {"p": Counter(), "s": Counter()} for p in pick_thresholds}
        for d in det_thresholds
    }
    breakdown = Counter()
    p_wrong_bins = Counter()
    s_wrong_bins = Counter()
    p_err_sum = 0.0
    p_tp_n = 0
    s_err_sum = 0.0
    s_tp_n = 0

    ref_pick = 0.3
    ref_det = 0.5

    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["x"], batch["t"])
        det_logits = outputs["det"]
        det_probs = torch.sigmoid(det_logits)
        if det_probs.dim() > 1:
            det_probs = det_probs.amax(dim=-1)

        p_probs = torch.sigmoid(outputs["p"])
        s_probs = torch.sigmoid(outputs["s"])
        p_pp, s_pp = apply_p_before_s_constraint(p_probs, s_probs, ref_pick)

        det_true = batch["det"] > 0.5
        bsz = batch["x"].size(0)

        for i in range(bsz):
            is_event = bool(det_true[i].item())
            for dth in det_thresholds:
                detected = bool((det_probs[i] >= dth).item())
                if is_event:
                    det_acc[dth]["tp" if detected else "fn"] += 1
                elif detected:
                    det_acc[dth]["fp"] += 1

            if not is_event:
                continue
            if batch["p_valid"][i] <= 0 and batch["s_valid"][i] <= 0:
                continue

            det_ok = bool((det_probs[i] >= ref_det).item())
            for head, probs, idx_k, valid_k, wrong_bins in [
                ("p", p_pp[i], "p_idx", "p_valid", p_wrong_bins),
                ("s", s_pp[i], "s_idx", "s_valid", s_wrong_bins),
            ]:
                if batch[valid_k][i] <= 0:
                    continue
                gt = int(batch[idx_k][i].item())
                peak, pred = float(probs.max().item()), int(probs.argmax().item())
                has_peak = peak >= ref_pick
                if not det_ok:
                    breakdown[f"{head}_fn_missed_det"] += 1
                elif has_peak and abs(pred - gt) <= tol:
                    breakdown[f"{head}_tp"] += 1
                    err = abs(idx_to_sec(pred, seq_len) - idx_to_sec(gt, seq_len))
                    if head == "p":
                        p_err_sum += err
                        p_tp_n += 1
                    else:
                        s_err_sum += err
                        s_tp_n += 1
                elif has_peak:
                    breakdown[f"{head}_fn_wrong_peak"] += 1
                    err = abs(idx_to_sec(pred, seq_len) - idx_to_sec(gt, seq_len))
                    if err <= 0.1:
                        wrong_bins["<=0.1s"] += 1
                    elif err <= 0.25:
                        wrong_bins["0.1-0.25s"] += 1
                    elif err <= 0.5:
                        wrong_bins["0.25-0.5s(outside_tol)"] += 1
                    elif err <= 1.0:
                        wrong_bins["0.5-1.0s"] += 1
                    else:
                        wrong_bins[">1.0s"] += 1
                else:
                    breakdown[f"{head}_fn_no_peak"] += 1

        for dth in det_thresholds:
            det_pred = det_probs >= dth
            for pth in pick_thresholds:
                p_use, s_use = apply_p_before_s_constraint(p_probs, s_probs, pth)
                for head, probs, idx_k, valid_k, store in [
                    ("p", p_use, "p_idx", "p_valid", pick_acc[dth][pth]["p"]),
                    ("s", s_use, "s_idx", "s_valid", pick_acc[dth][pth]["s"]),
                ]:
                    peak, pred_idx = probs.max(dim=-1)
                    for i in range(bsz):
                        is_event = bool(det_true[i].item())
                        is_noise = not is_event
                        has_gt = bool(batch[valid_k][i].item())
                        detected = bool(det_pred[i].item())
                        peak_i = float(peak[i].item())
                        pred_exists = peak_i >= pth
                        pred_i = int(pred_idx[i].item())
                        if is_noise:
                            if pred_exists:
                                store["fp"] += 1
                            continue
                        if not has_gt:
                            continue
                        gt_i = int(batch[idx_k][i].item())
                        within = pred_exists and abs(pred_i - gt_i) <= tol
                        if not detected:
                            store["fn"] += 1
                        elif within:
                            store["tp"] += 1
                        else:
                            store["fn"] += 1

        if (bi + 1) % 200 == 0:
            print(f"[analysis] batch {bi+1}/{len(loader)}", flush=True)

    det_sweep = {
        str(d): pick_prf(v["tp"], v["fp"], v["fn"])
        for d, v in det_acc.items()
    }
    pick_sweep = {}
    for dth in det_thresholds:
        pick_sweep[str(dth)] = {}
        for pth in pick_thresholds:
            p = pick_acc[dth][pth]["p"]
            s = pick_acc[dth][pth]["s"]
            pick_sweep[str(dth)][str(pth)] = {
                "p_f1": pick_prf(p["tp"], p["fp"], p["fn"])["f1"],
                "s_f1": pick_prf(s["tp"], s["fp"], s["fn"])["f1"],
                "p_recall": pick_prf(p["tp"], p["fp"], p["fn"])["recall"],
                "s_recall": pick_prf(s["tp"], s["fp"], s["fn"])["recall"],
            }

    ref_det_m = det_sweep[str(ref_det)]
    p_counts = pick_acc[ref_det][ref_pick]["p"]
    s_counts = pick_acc[ref_det][ref_pick]["s"]
    p_m = pick_prf(p_counts["tp"], p_counts["fp"], p_counts["fn"])
    s_m = pick_prf(s_counts["tp"], s_counts["fp"], s_counts["fn"])

    return {
        "checkpoint": str(checkpoint),
        "n_test": len(ds),
        "default": {
            "det_threshold": ref_det,
            "pick_threshold": ref_pick,
            "det": ref_det_m,
            "p": p_m,
            "s": s_m,
        },
        "error_breakdown_pick03_det05": dict(breakdown),
        "p_wrong_peak_distance": dict(p_wrong_bins),
        "s_wrong_peak_distance": dict(s_wrong_bins),
        "p_mae_sec": p_err_sum / max(p_tp_n, 1),
        "s_mae_sec": s_err_sum / max(s_tp_n, 1),
        "det_threshold_sweep": det_sweep,
        "pick_threshold_sweep": pick_sweep,
    }


def parse_threshold_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("threshold list must not be empty")
    return vals


def export_pick_curve(
    report: dict,
    det_threshold: float = 0.5,
) -> list[dict]:
    det_key = str(det_threshold)
    det_f1 = report["det_threshold_sweep"][det_key]["f1"]
    rows = []
    for pick_key, metrics in sorted(
        report["pick_threshold_sweep"][det_key].items(),
        key=lambda kv: float(kv[0]),
    ):
        rows.append(
            {
                "pick_threshold": float(pick_key),
                "det_f1": det_f1,
                "p_f1": metrics["p_f1"],
                "s_f1": metrics["s_f1"],
                "p_recall": metrics["p_recall"],
                "s_recall": metrics["s_recall"],
            }
        )
    return rows


def plot_pick_curves(curve_rows: list[dict], output_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    xs = [r["pick_threshold"] for r in curve_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [r["det_f1"] for r in curve_rows], marker="o", label="det F1")
    ax.plot(xs, [r["p_f1"] for r in curve_rows], marker="o", label="P F1")
    ax.plot(xs, [r["s_f1"] for r in curve_rows], marker="o", label="S F1")
    ax.axhline(0.995, color="C0", linestyle="--", alpha=0.35, linewidth=1)
    ax.axhline(0.95, color="C1", linestyle="--", alpha=0.35, linewidth=1)
    ax.axhline(0.95, color="C2", linestyle="--", alpha=0.35, linewidth=1)
    ax.set_xlabel("pick threshold")
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.set_xlim(min(xs) - 0.02, max(xs) + 0.02)
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/run14/14_main/best.pt")
    p.add_argument("--output", default="outputs/run14/14_main/analysis.json")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument(
        "--pick-thresholds",
        default="0.20,0.25,0.28,0.30,0.32,0.35",
        help="comma-separated pick thresholds",
    )
    p.add_argument(
        "--det-thresholds",
        default="0.40,0.45,0.50,0.55",
        help="comma-separated det thresholds",
    )
    p.add_argument(
        "--curve-det-threshold",
        type=float,
        default=0.5,
        help="det threshold used when exporting/plotting pick curves",
    )
    p.add_argument(
        "--curve-json",
        default="",
        help="optional flat JSON path for pick-threshold curves",
    )
    p.add_argument(
        "--curve-png",
        default="",
        help="optional PNG path for pick-threshold curves",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = analyze(
        Path(args.checkpoint),
        args.seq_len,
        device,
        pick_thresholds=parse_threshold_list(args.pick_thresholds),
        det_thresholds=parse_threshold_list(args.det_thresholds),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    curve_rows = export_pick_curve(report, det_threshold=args.curve_det_threshold)
    if args.curve_json:
        curve_out = Path(args.curve_json)
        curve_out.parent.mkdir(parents=True, exist_ok=True)
        curve_out.write_text(json.dumps(curve_rows, indent=2))
    if args.curve_png:
        plot_pick_curves(
            curve_rows,
            Path(args.curve_png),
            title=f"pick threshold sweep (det={args.curve_det_threshold})",
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
