#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full STEAD test comparison: default denoise split vs bypass (no denoise)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analyze_stead_picking import analyze, export_pick_curve, plot_pick_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/realtime_full_compare")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint)
    pick_th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    det_th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    report = {"checkpoint": str(ckpt), "modes": {}}
    for mode, bypass in [("default_denoise_split", False), ("bypass_no_denoise", True)]:
        print(f"\n=== full test: {mode} ===", flush=True)
        sub = out_dir / mode
        sub.mkdir(exist_ok=True)
        r = analyze(ckpt, args.seq_len, device, pick_th, det_th, bypass_noise_cancel=bypass)
        (sub / "analysis.json").write_text(json.dumps(r, indent=2))
        curve = export_pick_curve(r, det_threshold=0.5)
        (sub / "threshold_sweep_curve.json").write_text(json.dumps(curve, indent=2))
        plot_pick_curves(curve, sub / "threshold_sweep_curve.png", title=f"{mode} (det=0.5)")
        report["modes"][mode] = {
            "bypass_noise_cancel": bypass,
            "n_test": r["n_test"],
            "default_det_f1": r["default"]["det"]["f1"],
            "default_p_f1": r["default"]["p"]["f1"],
            "default_s_f1": r["default"]["s"]["f1"],
            "error_breakdown": r["error_breakdown_pick03_det05"],
        }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[full_compare] -> {out_dir}")


if __name__ == "__main__":
    main()
