# -*- coding: utf-8 -*-
"""Direct-ray path extraction helpers for 1D layered models."""

from __future__ import annotations

import torch

from hnf.inversion_1d import LayeredEarth1D, find_ray_parameter, layer_index_at_depth


def direct_ray_path(
    model: LayeredEarth1D,
    phase: str,
    source_depth: float | torch.Tensor,
    receiver_distance: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return piecewise-linear direct-ray path as (x, z) nodes.

    The source is placed at x=0, z=source_depth and the receiver at x=offset, z=0.
    """
    v = model.velocity(phase)
    dev = v.device
    dtype = v.dtype
    src_z = source_depth if isinstance(source_depth, torch.Tensor) else torch.tensor(source_depth, device=dev, dtype=dtype)
    rx = receiver_distance if isinstance(receiver_distance, torch.Tensor) else torch.tensor(receiver_distance, device=dev, dtype=dtype)
    if src_z.dim() != 0 or rx.dim() != 0:
        raise ValueError("direct_ray_path expects scalar source_depth and receiver_distance")

    if float(abs(rx).item()) < 1e-8:
        return torch.tensor([0.0, 0.0], device=dev, dtype=dtype), torch.stack([src_z, torch.tensor(0.0, device=dev, dtype=dtype)])

    p = find_ray_parameter(model.depths, v, src_z.unsqueeze(0), rx.unsqueeze(0))[0]
    layer_idx = int(layer_index_at_depth(model.depths, src_z).item())

    x_nodes = [torch.tensor(0.0, device=dev, dtype=dtype)]
    z_nodes = [src_z]
    x_curr = torch.tensor(0.0, device=dev, dtype=dtype)
    z_curr = src_z

    for i in range(layer_idx, -1, -1):
        z_top = model.depths[i]
        dz = z_curr - z_top
        if float(dz.item()) <= 0.0:
            z_curr = z_top
            continue
        pv = p * v[i]
        cos_i = torch.sqrt((1.0 - pv * pv).clamp(min=1e-8))
        dx = dz * pv * v[i] / cos_i
        x_curr = x_curr + dx
        z_curr = z_top
        x_nodes.append(x_curr)
        z_nodes.append(z_curr)

    x = torch.stack(x_nodes)
    z = torch.stack(z_nodes)
    if x.numel() >= 1:
        x[-1] = rx
    return x, z
