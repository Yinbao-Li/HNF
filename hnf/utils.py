# -*- coding: utf-8 -*-
"""Part 9: utility helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def generate_wave_data(
    n_points: int = 1000,
    n_sources: int = 5,
    dim: int = 2,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    sources = torch.randn(n_sources, dim) * 5
    x = torch.randn(n_points, dim) * 8
    field = torch.zeros(n_points, 1)
    for s in sources:
        dist = torch.norm(x - s, dim=1, keepdim=True)
        field += torch.sin(2 * dist) / (dist + 0.1)
    field += noise_std * torch.randn(n_points, 1)
    return x, sources, field


def plot_kernel_matrix(
    k: torch.Tensor,
    title: str = "惠更斯核矩阵",
    save_path: Optional[str] = None,
):
    k_np = torch.abs(k.squeeze(0)).numpy() if k.dim() == 3 else torch.abs(k).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow(k_np, cmap="hot", interpolation="nearest")
    axes[0].set_title(f"{title}\n(热力图)")
    plt.colorbar(im, ax=axes[0])
    row_idx = len(k_np) // 2
    axes[1].plot(k_np[row_idx, :])
    axes[1].grid(True)
    eigvals = np.linalg.eigvalsh(k_np + 1e-6 * np.eye(len(k_np)))
    axes[2].hist(eigvals, bins=30, color="green", alpha=0.7)
    axes[2].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    return_dict: bool = True,
) -> Union[Dict[str, float], Tuple[float, float]]:
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()
    if return_dict:
        return {"MSE": mse, "MAE": mae}
    return mse, mae
