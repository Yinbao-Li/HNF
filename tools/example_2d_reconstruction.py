#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D field reconstruction demo: synthetic data → HNF → visualization.

Run from the HNF project root:
    python example_2d_reconstruction.py
    python example_2d_reconstruction.py --field-type vortex --n-obs 200 --train-steps 300"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

import torch

from hnf.data_generator import build_synthetic_sample
from hnf.field import HuygensNeuralField
from hnf.visualize import plot_observation_distribution, plot_reconstruction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF 2D sparse field reconstruction demo")
    p.add_argument("--field-type", default="plane_wave", choices=["plane_wave", "radial_wave", "vortex", "mixed"])
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--n-obs", type=int, default=128)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--omega", type=float, default=6.2831853)
    p.add_argument("--train-steps", type=int, default=200, help="Steps to fine-tune gamma/omega")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-density", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HNF] device={device}, field={args.field_type}, n_obs={args.n_obs}")

    sample = build_synthetic_sample(
        field_type=args.field_type,
        resolution=args.resolution,
        n_obs=args.n_obs,
        seed=args.seed,
    )

    obs_coords = sample.obs_coords.to(device)
    obs_values = sample.obs_values.to(device)
    target_coords = sample.grid_coords.to(device)
    target_values = sample.field_values.to(device)

    model = HuygensNeuralField(
        gamma=args.gamma,
        omega=args.omega,
        alpha=args.alpha,
        eps=args.eps,
        learnable_gamma=True,
        learnable_omega=True,
        use_density=args.use_density,
        causal=False,
    ).to(device)

    if args.train_steps > 0:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for step in range(1, args.train_steps + 1):
            opt.zero_grad()
            pred = model(obs_coords, obs_values, target_coords)
            loss = torch.mean((pred - target_values) ** 2)
            loss.backward()
            opt.step()
            if step % max(1, args.train_steps // 5) == 0 or step == 1:
                print(
                    f"  step {step:4d}/{args.train_steps}  mse={loss.item():.6f}  "
                    f"gamma={model.gamma.item():.4f}  omega={model.omega.item():.4f}"
                )

    with torch.no_grad():
        field_pred = model(obs_coords, obs_values, target_coords)

    mae = (field_pred - target_values).abs().mean().item()
    rmse = torch.sqrt(torch.mean((field_pred - target_values) ** 2)).item()
    print(f"[HNF] reconstruction MAE={mae:.6f}  RMSE={rmse:.6f}")

    plot_observation_distribution(
        sample.obs_coords,
        resolution=sample.resolution,
        save_path=out_dir / f"obs_{args.field_type}.png",
    )
    plot_reconstruction(
        sample,
        field_pred.cpu(),
        save_path=out_dir / f"reconstruction_{args.field_type}.png",
    )
    print(f"[HNF] figures saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
