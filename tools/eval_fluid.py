#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate fluid Stage-0 reconstructor on held-out synthetic test set."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hnf.fluid_dataset import SyntheticFluidDataset
from hnf.fluid_model import FluidHNFReconstructor
from tools.train_fluid import evaluate, rel_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval fluid Stage-0")
    p.add_argument("--checkpoint", default="outputs/fluid/stage0_synth/best.pt")
    p.add_argument("--output", default="outputs/fluid/stage0_synth/test_metrics.json")
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
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
    h = int(a.get("h", 32))
    w = int(a.get("w", 32))
    keep = float(a.get("keep_frac", 0.1))
    predict_eta = not bool(a.get("no_eta", False))

    ds = SyntheticFluidDataset(
        split="test",
        n_samples=args.n_test,
        h=h,
        w=w,
        keep_frac=keep,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = FluidHNFReconstructor(
        h=h,
        w=w,
        embed_dim=int(a.get("embed_dim", 64)),
        dropout=float(a.get("dropout", 0.1)),
        principle=str(a.get("principle", "huygens_fresnel")),
        predict_eta=predict_eta,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    metrics = evaluate(model, loader, device, float(a.get("eta_weight", 0.1)))

    # Per-family breakdown
    by_fam: dict[str, list[float]] = {}
    for batch in loader:
        x = batch["x"].to(device)
        dense = batch["dense"].to(device)
        pred = model(x)
        for i, fam in enumerate(batch["family"]):
            by_fam.setdefault(str(fam), []).append(rel_err(pred[i], dense[i]))

    result = {
        "checkpoint": str(args.checkpoint),
        "n_test": args.n_test,
        "keep_frac": keep,
        "vel_mse": metrics["vel_mse"],
        "vel_rel": metrics["vel_rel"],
        "eta_mse": metrics["eta_mse"],
        "eta_rel": metrics["eta_rel"],
        "vel_rel_by_family": {k: float(np.mean(v)) for k, v in by_fam.items()},
        "n_params": int(ckpt.get("n_params", -1)),
        "kernel_params": ckpt.get("kernel_params", {}),
        "note": "Synthetic 2D Stage-0; RACLETTE .pv loader deferred (needs pyvista).",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("vel_rel", "eta_rel", "vel_rel_by_family")}, indent=2))
    print(f"[fluid-eval] wrote {out}")


if __name__ == "__main__":
    main()
