#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate RACLETTE Stage-0b reconstructor."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hnf.fluid_model import FluidHNFReconstructor
from hnf.raclette_dataset import RacletteSliceDataset
from tools.train_fluid import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval RACLETTE Stage-0b")
    p.add_argument("--checkpoint", default="outputs/fluid/stage0b_raclette/best.pt")
    p.add_argument("--cache", default="external_data/raclette_cache/gt_slices.npz")
    p.add_argument("--output", default="outputs/fluid/stage0b_raclette/test_metrics.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    keep = float(a.get("keep_frac", 0.1))
    ds = RacletteSliceDataset(args.cache, "test", keep, args.seed, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    h, w = int(ds.velocity.shape[2]), int(ds.velocity.shape[3])
    model = FluidHNFReconstructor(
        h=h,
        w=w,
        embed_dim=int(a.get("embed_dim", 64)),
        dropout=float(a.get("dropout", 0.1)),
        principle=str(a.get("principle", "huygens_fresnel")),
        predict_eta=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    metrics = evaluate(model, loader, device, eta_weight=0.0)
    result = {
        "checkpoint": str(args.checkpoint),
        "n_test": len(ds),
        "keep_frac": keep,
        "vel_mse": metrics["vel_mse"],
        "vel_rel": metrics["vel_rel"],
        "n_params": int(ckpt.get("n_params", -1)),
        "kernel_params": ckpt.get("kernel_params", {}),
        "note": "RACLETTE GT in-plane slices; artificial sparsity (not MRI→CFD yet).",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("vel_rel", "vel_mse", "n_test")}, indent=2))
    print(f"[raclette-eval] wrote {out}")


if __name__ == "__main__":
    main()
