#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export run20 picking prior (rho, picks, kernel) without running inversion."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from hnf.inversion_1d import default_station_distances, default_synth_model, synthesize_travel_times
from hnf.picking_prior import build_picking_prior, load_picking_model_from_checkpoint, save_prior_cache
from hnf.synth_waveforms_1d import synthesize_multistation_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export run20 picking prior cache")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output", default="outputs/inv06_run20_prior/prior_cache.json")
    p.add_argument("--n-stations", type=int, default=8)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--trace-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infer-seq-len", type=int, default=None, help="Optional downsample (may break det head if != train seq)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[export] loading {ckpt} on {device}...", flush=True)
    model, ckpt_args = load_picking_model_from_checkpoint(ckpt, device, bypass=True)
    seq_len = int(ckpt_args.get("seq_len", 800))

    true_model = default_synth_model(device)
    distances = default_station_distances(device, args.n_stations)
    clean = synthesize_travel_times(true_model, args.source_depth, distances)
    x, t, _ = synthesize_multistation_batch(
        true_model, args.source_depth, distances,
        seq_len=seq_len, noise_std=args.trace_noise, seed=args.seed + 10,
    )
    x, t = x.to(device), t.to(device)

    print("[export] running picking (one station at a time)...", flush=True)
    infer_len = args.infer_seq_len if args.infer_seq_len and args.infer_seq_len > 0 else 600
    prior = build_picking_prior(
        model, x, t, true_model, clean["tp"], clean["ts"],
        vp_perturb_seed=args.seed + 1,
        infer_seq_len=infer_len,
    )
    out = Path(args.output)
    save_prior_cache(prior, out)
    print(
        f"[export] pick MAE P={prior.pick_mae_p:.4f}s S={prior.pick_mae_s:.4f}s "
        f"kernel vp/vs={prior.kernel_vp:.2f}/{prior.kernel_vs:.2f} -> {out}",
        flush=True,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
