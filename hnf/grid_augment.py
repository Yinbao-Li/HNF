# -*- coding: utf-8 -*-
"""Temporal-grid augmentation: same 60 s window, different sample rates.

The run28 picking model is grid-locked: trained only at 800 points over 60 s
(13.3 Hz), its detection collapses when the same window is fed at 400 or 200
points (det AUC 0.997 -> 0.85, event det median 0.96 -> 0.000). That single
weakness blocks both the 100 Hz / 6000-point fine-tune and any coarse-to-fine
inference speedup, since every cheap tier needs a different grid.

This module resamples a whole batch onto another grid and carries every time
label along with it, so training can mix sample rates within one run.

Index convention follows the dataset and the metrics (``idx * 60 / seq_len``,
not ``seq_len - 1``); mixing the two conventions shifts picks by up to one bin.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from hnf.stead_picking_dataset import gaussian_pick_label


def rescale_index(idx: torch.Tensor, in_len: int, out_len: int) -> torch.Tensor:
    """Move pick indices to a new grid, preserving the -1 'absent' sentinel."""
    out = torch.round(idx.float() * float(out_len) / float(in_len))
    out = out.clamp(0, out_len - 1).long()
    return torch.where(idx < 0, torch.full_like(out, -1), out)


def resample_batch_to_grid(
    batch: dict[str, torch.Tensor],
    out_len: int,
    *,
    window_sec: float = 60.0,
    label_sigma_sec: float = 0.4,
) -> dict[str, torch.Tensor]:
    """Return a copy of ``batch`` living on an ``out_len``-point grid.

    Pick targets are rebuilt as fresh Gaussians rather than interpolated, which
    would flatten the peak and weaken the label as the grid gets coarser.
    """
    x = batch["x"]
    in_len = x.size(1)
    if in_len == out_len:
        return batch

    out = dict(batch)
    out["x"] = F.interpolate(
        x.transpose(1, 2).float(), size=out_len, mode="linear", align_corners=False
    ).transpose(1, 2).to(x.dtype)
    out["t"] = (
        torch.linspace(0.0, window_sec, out_len, device=x.device, dtype=x.dtype)
        .view(1, out_len, 1)
        .expand(x.size(0), out_len, 1)
        .contiguous()
    )

    sigma_samples = max(1.0, label_sigma_sec * out_len / window_sec)
    for idx_key, tgt_key in (("p_idx", "p_target"), ("s_idx", "s_target")):
        if idx_key not in batch:
            continue
        new_idx = rescale_index(batch[idx_key], in_len, out_len)
        out[idx_key] = new_idx
        if tgt_key in batch:
            out[tgt_key] = torch.stack(
                [
                    gaussian_pick_label(int(i.item()), out_len, sigma_samples)
                    for i in new_idx
                ]
            ).to(device=x.device, dtype=batch[tgt_key].dtype)
    return out


def parse_grid_lens(spec: Optional[str]) -> list[int]:
    """Parse a '400,800,1200' style CLI spec into sorted unique lengths."""
    if not spec:
        return []
    lens = {int(part) for part in str(spec).replace(" ", "").split(",") if part}
    if any(v < 16 for v in lens):
        raise ValueError(f"grid lengths must be >= 16, got {sorted(lens)}")
    return sorted(lens)


def sample_grid_len(
    lens: Sequence[int],
    *,
    prob: float,
    base_len: int,
    rng: Optional[random.Random] = None,
) -> int:
    """Pick a grid length for this step; ``base_len`` when the draw misses."""
    if not lens or prob <= 0.0:
        return base_len
    r = rng if rng is not None else random
    if r.random() >= prob:
        return base_len
    return r.choice(list(lens))


def iter_grid_lens(lens: Iterable[int], base_len: int) -> list[int]:
    """Validation grids, always including the native one for comparison."""
    out = sorted({int(v) for v in lens} | {int(base_len)})
    return out
