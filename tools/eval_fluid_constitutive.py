#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate fluid Stage-1 constitutive recovery on held-out synthetic test."""

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

from hnf.fluid_constitutive_model import (
    CONST_ID_TO_FAMILY,
    ConstitutiveFluidDataset,
    FluidConstitutiveModel,
)
from tools.train_fluid_constitutive import evaluate, rel_err_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval fluid Stage-1 constitutive")
    p.add_argument("--checkpoint", default="outputs/fluid/stage1_constitutive/best.pt")
    p.add_argument("--output", default="outputs/fluid/stage1_constitutive/test_metrics.json")
    p.add_argument("--n-test", type=int, default=1024)
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
    h, w = int(a.get("h", 32)), int(a.get("w", 32))
    keep = float(a.get("keep_frac", 0.1))

    ds = ConstitutiveFluidDataset("test", args.n_test, h, w, keep, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = FluidConstitutiveModel(
        h=h,
        w=w,
        embed_dim=int(a.get("embed_dim", 64)),
        dropout=float(a.get("dropout", 0.1)),
        principle=str(a.get("principle", "huygens_fresnel")),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    metrics = evaluate(model, loader, device, argparse.Namespace())

    # Per-family η0 / θ breakdown
    by_fam: dict[str, dict[str, list[float]]] = {
        "newtonian": {"eta_rel": [], "theta_rel": [], "vel_rel": []},
        "carreau": {"eta_rel": [], "theta_rel": [], "vel_rel": []},
    }
    cm = np.zeros((2, 2), dtype=np.int64)
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        dense = batch["dense"].to(device)
        y = torch.as_tensor(batch["family_id"], device=device, dtype=torch.long)
        pred = out["family_logits"].argmax(-1)
        theta_t = batch["theta"].to(device)
        mask = batch["theta_mask"].to(device)
        for i in range(x.size(0)):
            yi, pi = int(y[i]), int(pred[i])
            cm[yi, pi] += 1
            fam = CONST_ID_TO_FAMILY[yi]
            gt0 = float(theta_t[i, 0])
            pr0 = float(out["theta"][i, 0])
            by_fam[fam]["eta_rel"].append(abs(pr0 - gt0) / max(abs(gt0), 1e-6))
            m = mask[i]
            num = ((out["theta"][i] - theta_t[i]) * m).pow(2).sum().sqrt()
            den = (theta_t[i].abs() * m).pow(2).sum().sqrt().clamp_min(1e-8)
            by_fam[fam]["theta_rel"].append(float((num / den).item()))
            by_fam[fam]["vel_rel"].append(rel_err_tensor(out["dense"][i], dense[i]))

    per_family = {
        fam: {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()}
        for fam, d in by_fam.items()
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "n_test": args.n_test,
        "keep_frac": keep,
        **metrics,
        "confusion_matrix": cm.tolist(),
        "per_family": per_family,
        "n_params": int(ckpt.get("n_params", -1)),
        "kernel_params": ckpt.get("kernel_params", {}),
        "id_to_family": CONST_ID_TO_FAMILY,
        "note": "Stage-1 constitutive synthetic (Newtonian vs Carreau); not RACLETTE GT.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                "family_acc": result["family_acc"],
                "eta_rel": result["eta_rel"],
                "theta_rel": result["theta_rel"],
                "vel_rel": result["vel_rel"],
                "per_family": per_family,
            },
            indent=2,
        )
    )
    print(f"[fluid-s1-eval] wrote {out}")


if __name__ == "__main__":
    main()
