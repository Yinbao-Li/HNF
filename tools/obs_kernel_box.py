#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers: freeze trunk and manipulate Huygens kernel box params (γ/ω/c)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


KERNEL_BOX_KEYS = ("gamma", "omega", "wave_speed", "c_log_scale", "obliquity_scale")


def iter_kernels(model: nn.Module) -> list[tuple[str, nn.Module]]:
    out: list[tuple[str, nn.Module]] = []
    if getattr(model, "multi_scale_encoder", None) is not None:
        for si, branch in enumerate(model.multi_scale_encoder.branches):
            for li, layer in enumerate(branch.layers):
                out.append((f"scale{si}_layer{li}", layer.kernel))
    for i, layer in enumerate(getattr(model, "p_layers", []) or []):
        out.append((f"p_branch_{i}", layer.kernel))
    for i, layer in enumerate(getattr(model, "s_layers", []) or []):
        out.append((f"s_branch_{i}", layer.kernel))
    nc = getattr(model, "noise_cancel_branch", None)
    if nc is not None and hasattr(nc, "prop_kernel"):
        out.append(("noise_prop", nc.prop_kernel))
    return out


def freeze_all_but_kernels(model: nn.Module, include_noise_kernel: bool = False) -> list[str]:
    """Freeze everything; unfreeze only HuygensKernel physical box params."""
    for p in model.parameters():
        p.requires_grad = False
    trained = []
    for name, k in iter_kernels(model):
        if (not include_noise_kernel) and name.startswith("noise_"):
            continue
        for key in KERNEL_BOX_KEYS:
            if not hasattr(k, key):
                continue
            tensor = getattr(k, key)
            if isinstance(tensor, nn.Parameter):
                tensor.requires_grad = True
                trained.append(f"{name}.{key}")
    return trained


def snapshot_kernel_state(model: nn.Module) -> dict[str, dict[str, float]]:
    state: dict[str, dict[str, float]] = {}
    for name, k in iter_kernels(model):
        entry = {}
        for key in KERNEL_BOX_KEYS:
            if hasattr(k, key):
                entry[key] = float(getattr(k, key).detach().cpu())
        if hasattr(k, "effective_gamma"):
            entry["eff_gamma"] = float(k.effective_gamma().detach().cpu())
            entry["eff_omega"] = float(k.effective_omega().detach().cpu())
            entry["eff_c"] = float(k.effective_wave_speed().detach().cpu())
        state[name] = entry
    return state


def restore_kernel_raw(model: nn.Module, state: dict[str, dict[str, float]]) -> None:
    with torch.no_grad():
        for name, k in iter_kernels(model):
            if name not in state:
                continue
            for key in KERNEL_BOX_KEYS:
                if key in state[name] and hasattr(k, key):
                    getattr(k, key).data.fill_(float(state[name][key]))


def softplus_inv(y: float) -> float:
    y = max(float(y), 1e-4)
    # softplus^{-1}(y) = log(exp(y)-1)
    return float(np.log(np.expm1(y)))


def apply_group_log_scales(
    model: nn.Module,
    base: dict[str, dict[str, float]],
    scales: dict[str, float],
) -> None:
    """Apply multiplicative scales on effective γ/ω/c via raw softplus params.

    scales keys e.g.:
      p_gamma, p_omega, p_c, s_gamma, s_omega, s_c,
      ms_gamma, ms_omega, ms_c  (multi-scale encoder)
    """
    with torch.no_grad():
        for name, k in iter_kernels(model):
            if name not in base:
                continue
            b = base[name]
            if name.startswith("p_"):
                g_s, o_s, c_s = scales.get("p_gamma", 1.0), scales.get("p_omega", 1.0), scales.get("p_c", 1.0)
            elif name.startswith("s_"):
                g_s, o_s, c_s = scales.get("s_gamma", 1.0), scales.get("s_omega", 1.0), scales.get("s_c", 1.0)
            elif name.startswith("scale"):
                g_s, o_s, c_s = scales.get("ms_gamma", 1.0), scales.get("ms_omega", 1.0), scales.get("ms_c", 1.0)
            else:
                continue
            # Restore raw then set to hit target effective ≈ base_eff * scale
            if "gamma" in b and hasattr(k, "gamma") and isinstance(k.gamma, nn.Parameter):
                target = max(1e-3, float(b.get("eff_gamma", 0.5)) * float(g_s))
                k.gamma.data.fill_(softplus_inv(target - 1e-3))
            if "omega" in b and hasattr(k, "omega") and isinstance(k.omega, nn.Parameter):
                target = max(1e-3, float(b.get("eff_omega", 0.3)) * float(o_s))
                k.omega.data.fill_(softplus_inv(target))
            if "c_log_scale" in b and hasattr(k, "c_log_scale") and isinstance(k.c_log_scale, nn.Parameter):
                # c_eff = c_base * exp(c_log_scale); fold scale into log_scale
                k.c_log_scale.data.fill_(float(b["c_log_scale"]) + float(np.log(max(1e-3, c_s))))


GROUP_SCALE_NAMES = (
    "p_gamma", "p_omega", "p_c",
    "s_gamma", "s_omega", "s_c",
    "ms_gamma", "ms_omega", "ms_c",
)


def scales_from_vector(vec: Iterable[float], names: tuple[str, ...] = GROUP_SCALE_NAMES) -> dict[str, float]:
    return {n: float(v) for n, v in zip(names, vec)}


def vector_from_scales(scales: dict[str, float], names: tuple[str, ...] = GROUP_SCALE_NAMES) -> np.ndarray:
    return np.array([float(scales.get(n, 1.0)) for n in names], dtype=np.float64)
