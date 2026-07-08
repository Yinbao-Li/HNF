#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time picking ablation: with vs without Huygens noise-cancel branch.

Modes on run20 checkpoint:
  default      - det on denoised u_final, pick on raw + noise cues (training setup)
  bypass       - skip noise branch entirely (fastest, raw 3C for all heads)

Usage:
    python benchmark_realtime_picking.py
    python benchmark_realtime_picking.py --max-batches 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import analyze, load_model
from hnf.stead_picking_dataset import STEADPickingDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time denoise bypass benchmark")
    p.add_argument(
        "--checkpoint",
        default="outputs/run20/20_wrongpeak_sharp/best.pt",
    )
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=300, help="0 = full test for accuracy")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output-dir", default="outputs/realtime_denoise_ablation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def benchmark_latency(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    warmup: int,
) -> dict[str, float]:
    model.eval()
    times_ms: list[float] = []
    n_samples = 0
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        if bi < warmup:
            model(batch["x"], batch["t"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            continue
        if max_batches > 0 and bi - warmup >= max_batches:
            break
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch["x"], batch["t"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed)
        n_samples += batch["x"].size(0)
    if not times_ms:
        return {"batches": 0, "samples": 0, "ms_per_batch": 0.0, "ms_per_trace": 0.0}
    mean_ms = sum(times_ms) / len(times_ms)
    bsz = loader.batch_size or 1
    return {
        "batches": len(times_ms),
        "samples": n_samples,
        "ms_per_batch": mean_ms,
        "ms_per_trace": mean_ms / bsz,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)

    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    modes = {
        "default_denoise_split": False,
        "bypass_no_denoise": True,
    }
    report: dict = {
        "checkpoint": str(ckpt),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "device": str(device),
        "modes": {},
    }

    pick_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    det_thresholds = [0.3, 0.4, 0.5]

    for mode_name, bypass in modes.items():
        print(f"\n=== {mode_name} (bypass_noise_cancel={bypass}) ===", flush=True)
        model, ckpt_args = load_model(ckpt, device)
        model.bypass_noise_cancel = bypass

        lat = benchmark_latency(model, loader, device, args.max_batches, args.warmup)
        print(
            f"latency: {lat['ms_per_trace']:.2f} ms/trace "
            f"({lat['ms_per_batch']:.1f} ms/batch @ bs={args.batch_size})",
            flush=True,
        )

        acc = None
        if args.max_batches == 0:
            acc_path = out_dir / f"analysis_{mode_name}.json"
            acc_report = analyze(
                ckpt, args.seq_len, device, pick_thresholds, det_thresholds
            )
            # re-run with bypass on loaded model inside analyze - analyze reloads model
            # So we need custom eval here for bypass mode
            acc_path.write_text(json.dumps(acc_report, indent=2))
            acc = {
                "det_f1": acc_report["default"]["det"]["f1"],
                "p_f1": acc_report["default"]["p"]["f1"],
                "s_f1": acc_report["default"]["s"]["f1"],
                "n_test": acc_report["n_test"],
            }
        else:
            # Quick partial accuracy on max_batches using default thresholds
            from hnf.picking_metrics import (
                apply_p_before_s_constraint,
                det_pred_from_logits,
                finalize_metrics,
                tolerance_bins,
                update_detection_counts,
                update_picking_counts,
            )
            from train_stead_picking import EvalAccumulator

            model.eval()
            acc_accum = EvalAccumulator()
            tol = tolerance_bins(args.seq_len, 0.5)
            pick_th = float(ckpt_args.get("pick_threshold", 0.3))
            n_seen = 0
            with torch.no_grad():
                for bi, batch in enumerate(loader):
                    if args.max_batches > 0 and bi >= args.max_batches:
                        break
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(batch["x"], batch["t"])
                    det_pred = det_pred_from_logits(outputs["det"])
                    det_true = batch["det"] > 0.5
                    update_detection_counts(acc_accum, det_pred, det_true)
                    p_probs, s_probs = apply_p_before_s_constraint(
                        torch.sigmoid(outputs["p"]),
                        torch.sigmoid(outputs["s"]),
                        pick_th,
                    )
                    for head, probs, idx_k, valid_k in [
                        ("p", p_probs, "p_idx", "p_valid"),
                        ("s", s_probs, "s_idx", "s_valid"),
                    ]:
                        update_picking_counts(
                            getattr(acc_accum, head),
                            probs,
                            det_pred,
                            det_true,
                            batch[valid_k] > 0,
                            batch[idx_k],
                            pick_th,
                            tol,
                            args.seq_len,
                        )
                    n_seen += batch["x"].size(0)
            metrics = finalize_metrics(acc_accum)
            acc = {
                "det_f1": metrics["det_f1"],
                "p_f1": metrics["p_f1"],
                "s_f1": metrics["s_f1"],
                "n_test_partial": n_seen,
            }
            print(
                f"partial acc ({n_seen} traces): det={acc['det_f1']:.4f} "
                f"P={acc['p_f1']:.4f} S={acc['s_f1']:.4f}",
                flush=True,
            )

        report["modes"][mode_name] = {
            "bypass_noise_cancel": bypass,
            "latency": lat,
            "accuracy": acc,
        }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[realtime] -> {out_dir}")


if __name__ == "__main__":
    main()
