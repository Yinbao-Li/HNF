#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate HNF on SST with SGF-RecFNO compatible metrics + FNO baseline comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "field" / "SGF-RecFNO"))

from benchmark.metrics import compute_field_metrics
from hnf.field import HuygensNeuralField
from hnf.sst_dataset import SSTDataset, SST_TEST, SST_VAL, SST_TRAIN, SST_STD


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HNF on SST (SGF-RecFNO metrics)")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--std", type=float, default=SST_STD)
    p.add_argument("--output", default=None, help="JSON output path")
    p.add_argument(
        "--fno-results",
        default=str(_ROOT.parent / "field" / "SGF-RecFNO" / "heat2D" / "logs" / "sst" / "comparison_results.json"),
        help="FNO baseline JSON for side-by-side comparison",
    )
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HuygensNeuralField(learnable_gamma=True, learnable_omega=True)
    state = ckpt["state_dict"]
    # Backward-compat: old checkpoints stored gamma/omega at top level
    if "gamma" in state and "kernel.gamma" not in state:
        state = {f"kernel.{k}" if k in ("gamma", "omega") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, ckpt


def split_indices(name: str):
    if name == "train":
        return SST_TRAIN
    if name == "val":
        return SST_VAL
    return SST_TEST


@torch.no_grad()
def run_eval(model, dataset, device, std: float):
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    ocean_mask = dataset.ocean_mask.to(device).unsqueeze(0).unsqueeze(0)
    preds, targets = [], []
    for batch in loader:
        obs_c = batch["obs_coords"][0].to(device)
        tgt_c = batch["target_coords"][0].to(device)
        obs_v = batch["obs_values"].to(device)
        batch_pred = []
        for b in range(obs_v.shape[0]):
            batch_pred.append(model(obs_c, obs_v[b], tgt_c).reshape(1, 180, 360))
        pred = torch.stack(batch_pred, dim=0)
        tgt = batch["field_2d"].to(device).unsqueeze(1)
        preds.append(pred * ocean_mask)
        targets.append(tgt * ocean_mask)
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(targets, dim=0)
    return compute_field_metrics(pred, tgt, std=std)


def load_fno_baselines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, ckpt = load_model(args.ckpt, device)
    ds = SSTDataset(split_indices(args.split))
    metrics = run_eval(model, ds, device, std=args.std)
    metrics.update({
        "model": "HNF",
        "split": args.split,
        "checkpoint": str(args.ckpt),
        "best_epoch": ckpt.get("epoch"),
    })

    out = Path(args.output) if args.output else Path(args.ckpt).parent / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print(f"\n=== HNF on SST {args.split} ===")
    for k in ("mae_k", "rmse_k", "relative_l2", "mse", "psnr", "ssim"):
        print(f"  {k:12s}: {metrics[k]:.6f}")

    baselines = load_fno_baselines(Path(args.fno_results))
    if baselines:
        print("\n=== FNO / RecFNO baselines (test) ===")
        for row in baselines:
            print(f"  {row['model']:18s}  mae_k={row['mae_k']:.4f}  rel_l2={row['relative_l2']:.4e}  ssim={row['ssim']:.4f}")
        best = min(baselines, key=lambda r: r["mae_k"])
        print(f"\n  Best FNO baseline: {best['model']}  mae_k={best['mae_k']:.4f}")
        print(f"  HNF gap: {metrics['mae_k'] - best['mae_k']:+.4f} K")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
