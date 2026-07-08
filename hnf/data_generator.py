# -*- coding: utf-8 -*-
"""Synthetic field data generators for HNF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

FieldType = Literal["plane_wave", "radial_wave", "vortex", "mixed"]


def _make_grid(
    resolution: int | tuple[int, int],
    domain: tuple[float, float] = (-1.0, 1.0),
    dim: int = 2,
) -> torch.Tensor:
    """Build a regular grid of coordinates in [-1, 1]^dim."""
    lo, hi = domain
    if dim == 2:
        if isinstance(resolution, int):
            ny = nx = resolution
        else:
            ny, nx = resolution
        ys = torch.linspace(lo, hi, ny)
        xs = torch.linspace(lo, hi, nx)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    if dim == 3:
        if isinstance(resolution, int):
            nz = ny = nx = resolution
        else:
            nz, ny, nx = resolution
        zs = torch.linspace(lo, hi, nz)
        ys = torch.linspace(lo, hi, ny)
        xs = torch.linspace(lo, hi, nx)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    raise ValueError(f"Unsupported spatial dimension: {dim}")


def generate_plane_wave(
    coords: torch.Tensor,
    direction: tuple[float, ...] = (1.0, 0.0),
    frequency: float = 2.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """u(x) = A * cos(2*pi*f * k·x + phase)."""
    k = torch.tensor(direction, dtype=coords.dtype, device=coords.device)
    k = k / (k.norm() + 1e-12)
    proj = coords @ k
    values = amplitude * torch.cos(2.0 * np.pi * frequency * proj + phase)
    return values.unsqueeze(-1)


def generate_radial_wave(
    coords: torch.Tensor,
    center: tuple[float, ...] = (0.0, 0.0),
    frequency: float = 3.0,
    amplitude: float = 1.0,
    decay: float = 0.5,
) -> torch.Tensor:
    """u(x) = A * sin(2*pi*f*r) / (r + eps) * exp(-decay*r)."""
    c = torch.tensor(center, dtype=coords.dtype, device=coords.device)
    r = (coords - c).norm(dim=-1)
    values = amplitude * torch.sin(2.0 * np.pi * frequency * r) / (r + 0.1)
    values = values * torch.exp(-decay * r)
    return values.unsqueeze(-1)


def generate_vortex_field(
    coords: torch.Tensor,
    center: tuple[float, float] = (0.0, 0.0),
    strength: float = 2.0,
    core_radius: float = 0.15,
) -> torch.Tensor:
    """Scalar vortex pattern: strength * atan2(dy, dx) * exp(-r^2 / core^2)."""
    if coords.shape[-1] < 2:
        raise ValueError("Vortex field requires at least 2D coordinates.")
    c = torch.tensor(center, dtype=coords.dtype, device=coords.device)
    d = coords[..., :2] - c
    angle = torch.atan2(d[..., 1], d[..., 0])
    r2 = (d ** 2).sum(dim=-1)
    values = strength * angle * torch.exp(-r2 / (core_radius ** 2 + 1e-12))
    return values.unsqueeze(-1)


def generate_field(
    field_type: FieldType,
    coords: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Dispatch to a specific synthetic field generator."""
    generators = {
        "plane_wave": generate_plane_wave,
        "radial_wave": generate_radial_wave,
        "vortex": generate_vortex_field,
    }
    if field_type == "mixed":
        parts = [
            generate_plane_wave(coords, frequency=1.5, amplitude=0.6),
            generate_radial_wave(coords, frequency=2.5, amplitude=0.5),
            generate_vortex_field(coords, strength=0.4),
        ]
        return sum(parts) / len(parts)
    return generators[field_type](coords, **kwargs)


def sample_sparse_observations(
    grid_coords: torch.Tensor,
    field_values: torch.Tensor,
    n_obs: int,
    strategy: Literal["random", "uniform", "latin"] = "random",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample sparse observation points from a dense grid field.

    Returns:
        obs_coords: [N, D]
        obs_values: [N, 1]
    """
    n_total = grid_coords.shape[0]
    n_obs = min(n_obs, n_total)
    rng = np.random.default_rng(seed)

    if strategy == "random":
        idx = rng.choice(n_total, size=n_obs, replace=False)
    elif strategy == "uniform":
        step = max(1, n_total // n_obs)
        idx = np.arange(0, n_total, step)[:n_obs]
    elif strategy == "latin":
        d = grid_coords.shape[1]
        bins = int(np.ceil(n_obs ** (1.0 / d)))
        lo = grid_coords.min(dim=0).values
        hi = grid_coords.max(dim=0).values
        samples = []
        for _ in range(n_obs * 4):
            u = lo + (hi - lo) * torch.rand(d)
            dist = ((grid_coords - u) ** 2).sum(dim=-1)
            samples.append(int(dist.argmin().item()))
            if len(set(samples)) >= n_obs:
                break
        idx = np.array(list(dict.fromkeys(samples))[:n_obs], dtype=np.int64)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    idx_t = torch.as_tensor(idx, dtype=torch.long)
    return grid_coords[idx_t], field_values[idx_t]


@dataclass
class SyntheticFieldSample:
    """Container for one synthetic reconstruction task."""

    grid_coords: torch.Tensor
    field_values: torch.Tensor
    obs_coords: torch.Tensor
    obs_values: torch.Tensor
    field_type: str
    resolution: int | tuple[int, int]


def build_synthetic_sample(
    field_type: FieldType = "plane_wave",
    resolution: int | tuple[int, int] = 64,
    n_obs: int = 128,
    dim: int = 2,
    sampling: Literal["random", "uniform", "latin"] = "random",
    seed: int | None = 42,
    **field_kwargs,
) -> SyntheticFieldSample:
    """Create a full synthetic sample: dense grid + sparse observations."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    grid_coords = _make_grid(resolution, dim=dim)
    field_values = generate_field(field_type, grid_coords, **field_kwargs)
    obs_coords, obs_values = sample_sparse_observations(
        grid_coords, field_values, n_obs=n_obs, strategy=sampling, seed=seed
    )
    return SyntheticFieldSample(
        grid_coords=grid_coords,
        field_values=field_values,
        obs_coords=obs_coords,
        obs_values=obs_values,
        field_type=field_type,
        resolution=resolution,
    )


class FieldDataset(Dataset):
    """PyTorch dataset of synthetic sparse-to-dense field reconstruction tasks."""

    def __init__(
        self,
        n_samples: int = 100,
        field_types: list[FieldType] | None = None,
        resolution: int = 64,
        n_obs: int = 128,
        dim: int = 2,
        seed: int = 0,
    ):
        self.samples: list[SyntheticFieldSample] = []
        types = field_types or ["plane_wave", "radial_wave", "vortex"]
        for i in range(n_samples):
            ftype = types[i % len(types)]
            sample = build_synthetic_sample(
                field_type=ftype,
                resolution=resolution,
                n_obs=n_obs,
                dim=dim,
                seed=seed + i,
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        return {
            "obs_coords": s.obs_coords,
            "obs_values": s.obs_values,
            "target_coords": s.grid_coords,
            "target_values": s.field_values,
            "field_type": s.field_type,
        }
