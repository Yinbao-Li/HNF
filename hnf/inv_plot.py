# -*- coding: utf-8 -*-
"""Shared plotting helpers for 1D inversion runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _stair(ax, depths_np, values, label, **kwargs):
    for i in range(len(values)):
        ax.plot(
            [float(values[i]), float(values[i])],
            [depths_np[i], depths_np[i + 1]],
            **kwargs,
        )
        if i < len(values) - 1 and values[i] != values[i + 1]:
            ax.plot(
                [float(values[i]), float(values[i + 1])],
                [depths_np[i + 1], depths_np[i + 1]],
                color=kwargs.get("color", "C0"),
                linestyle=kwargs.get("linestyle", "-"),
                alpha=kwargs.get("alpha", 1.0),
            )
    ax.plot([], [], label=label, **kwargs)


def plot_velocity_profiles(
    depths: torch.Tensor,
    true_vp: torch.Tensor,
    true_vs: torch.Tensor,
    init_vp: torch.Tensor,
    init_vs: torch.Tensor,
    rec_vp: torch.Tensor,
    rec_vs: torch.Tensor,
    out_path: Path,
    title: str = "1D layered inversion: true vs init vs recovered",
) -> None:
    d = depths.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, true_v, init_v, rec_v, xlab in [
        (axes[0], true_vp, init_vp, rec_vp, "Vp (km/s)"),
        (axes[1], true_vs, init_vs, rec_vs, "Vs (km/s)"),
    ]:
        _stair(ax, d, true_v.detach().cpu().numpy(), "true", color="black", linewidth=2)
        _stair(ax, d, init_v.detach().cpu().numpy(), "init", color="C1", linestyle="--")
        _stair(ax, d, rec_v.detach().cpu().numpy(), "recovered", color="C2", linestyle="-.")
        ax.set_xlabel(xlab)
        ax.set_ylabel("depth (km)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_q_profile(
    depths: torch.Tensor,
    true_q: torch.Tensor,
    init_q: torch.Tensor,
    rec_q: torch.Tensor,
    out_path: Path,
) -> None:
    d = depths.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    _stair(ax, d, true_q.detach().cpu().numpy(), "true", color="black", linewidth=2)
    _stair(ax, d, init_q.detach().cpu().numpy(), "init", color="C1", linestyle="--")
    _stair(ax, d, rec_q.detach().cpu().numpy(), "recovered", color="C2", linestyle="-.")
    ax.set_xlabel("Q")
    ax.set_ylabel("depth (km)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_misfit(history: list[dict[str, float]], out_path: Path, extra_keys: list[str] | None = None) -> None:
    steps = range(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["loss", "loss_tp", "loss_ts"]
    if extra_keys:
        keys.extend(extra_keys)
    for key in keys:
        if key in history[0]:
            ax.plot(steps, [h[key] for h in history], label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("misfit")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def perturb_initial(
    true_vp: torch.Tensor,
    true_vs: torch.Tensor,
    true_q: torch.Tensor,
    seed: int,
    q_scale: float = 1.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vp_scale = 1.08 + 0.02 * torch.randn((), generator=gen)
    vs_scale = 0.92 + 0.02 * torch.randn((), generator=gen)
    jitter = 0.03 * torch.randn(true_vp.shape, generator=gen)
    jitter = torch.cumsum(jitter, dim=0)
    vp0 = (true_vp.cpu() * vp_scale + jitter).clamp(min=1.5)
    for i in range(1, vp0.numel()):
        vp0[i] = torch.maximum(vp0[i], vp0[i - 1] + 0.05)
    vs0 = (true_vs.cpu() * vs_scale).clamp(min=1.0)
    for i in range(vs0.numel()):
        vs0[i] = torch.minimum(vs0[i], vp0[i] * 0.75)
    q0 = (true_q.cpu() * q_scale).clamp(min=20.0)
    for i in range(1, q0.numel()):
        q0[i] = torch.maximum(q0[i], q0[i - 1] + 5.0)
    return (
        vp0.to(true_vp.device),
        vs0.to(true_vs.device),
        q0.to(true_q.device),
    )
