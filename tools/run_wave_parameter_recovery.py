#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave-kernel (γ, ω, c) parameter recovery on synthetic impulse responses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
import sys

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.kernel import HuygensKernel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/wave_parameter_recovery")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=400)
    return p.parse_args()


def impulse_response(k: HuygensKernel, t: torch.Tensor) -> torch.Tensor:
    """Causal Green response of a unit impulse at t=0 on a 1-D time grid."""
    # Build lag matrix and evaluate amplitude×phase on positive lags from source 0.
    # Use kernel's own compute path: apply to one-hot source.
    B, T = 1, t.numel()
    src = torch.zeros(B, T, 1, device=t.device, dtype=t.dtype)
    src[:, 0, :] = 1.0
    # Dense time-lag kernel application (small T).
    dt = t.view(1, T, 1) - t.view(1, 1, T)
    lag = dt.clamp_min(0.0)
    # Mimic Fresnel/Huygens amplitude in time mode.
    gamma = k.effective_gamma()
    omega = k.effective_omega()
    c = k.effective_wave_speed()
    # Causal support: lag <= local window if set; mask by light-cone lag*c roughly lag itself in time mode.
    amp = torch.exp(-gamma * lag) / (lag + k.eps)
    if k.local_window_sec is not None:
        amp = amp * (lag <= float(k.local_window_sec) + 1e-6).to(amp.dtype)
    phase = torch.cos(omega * lag)
    K = amp * phase
    # Causal: only j<=i
    K = K * (dt >= 0).to(K.dtype)
    y = torch.einsum("bij,bjc->bic", K, src).squeeze(0).squeeze(-1)
    return y


def fit_from_target(
    y_tgt: torch.Tensor,
    t: torch.Tensor,
    steps: int,
    init: dict,
) -> dict:
    k = HuygensKernel(
        gamma=float(init["gamma"]),
        omega=float(init["omega"]),
        wave_speed=float(init["c"]),
        learnable_gamma=True,
        learnable_omega=True,
        learnable_wave_speed=True,
        distance_mode="time",
        local_window_sec=15.0,
        principle="huygens_fresnel",
    )
    # Reparameterize init near softplus inverses for stability.
    opt = torch.optim.Adam(
        [k.gamma, k.omega, k.wave_speed, k.c_log_scale],
        lr=0.05,
    )
    for _ in range(steps):
        opt.zero_grad()
        y = impulse_response(k, t)
        loss = F.mse_loss(y, y_tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return {
            "gamma": float(k.effective_gamma()),
            "omega": float(k.effective_omega()),
            "c": float(k.effective_wave_speed()),
            "loss": float(loss.detach()),
        }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t = torch.linspace(0.0, 8.0, 160)

    rows = []
    for i in range(int(args.n_trials)):
        g_true = float(rng.uniform(0.2, 2.0))
        w_true = float(rng.uniform(0.5, 4.0))
        c_true = float(rng.uniform(2.0, 10.0))
        k_true = HuygensKernel(
            gamma=g_true,
            omega=w_true,
            wave_speed=c_true,
            learnable_gamma=False,
            learnable_omega=False,
            learnable_wave_speed=False,
            distance_mode="time",
            local_window_sec=15.0,
            principle="huygens_fresnel",
        )
        with torch.no_grad():
            y = impulse_response(k_true, t)
            y = y + 0.02 * torch.randn_like(y)
        # Start from displaced init
        init = {
            "gamma": float(np.clip(g_true * rng.uniform(0.5, 1.5), 0.05, 5.0)),
            "omega": float(np.clip(w_true * rng.uniform(0.5, 1.5), 0.1, 8.0)),
            "c": float(np.clip(c_true * rng.uniform(0.5, 1.5), 0.5, 15.0)),
        }
        hat = fit_from_target(y.detach(), t, args.steps, init)
        rows.append(
            {
                "trial": i,
                "gamma_true": g_true,
                "omega_true": w_true,
                "c_true": c_true,
                "gamma_hat": hat["gamma"],
                "omega_hat": hat["omega"],
                "c_hat": hat["c"],
                "rel_err_gamma": abs(hat["gamma"] - g_true) / g_true,
                "rel_err_omega": abs(hat["omega"] - w_true) / w_true,
                "rel_err_c": abs(hat["c"] - c_true) / c_true,
                "loss": hat["loss"],
            }
        )

    def summ(key):
        v = np.asarray([r[key] for r in rows], float)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}

    report = {
        "n_trials": int(args.n_trials),
        "summary": {
            "rel_err_gamma": summ("rel_err_gamma"),
            "rel_err_omega": summ("rel_err_omega"),
            "rel_err_c": summ("rel_err_c"),
        },
        "rows": rows,
        "note": (
            "c is the causal support / light-cone scale in the production kernel "
            "(wave_speed parameter), recovered from impulse responses — not claimed "
            "as an identified medium velocity on field data."
        ),
    }
    (out / "RECOVERY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Wrote", out / "RECOVERY.json")


if __name__ == "__main__":
    main()
