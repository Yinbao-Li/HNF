# -*- coding: utf-8 -*-
"""Visualization utilities for HNF field reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _reshape_field(
    values: torch.Tensor | np.ndarray,
    resolution: int | tuple[int, int],
) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = _to_numpy(values)
    if isinstance(resolution, int):
        ny = nx = resolution
    else:
        ny, nx = resolution
    return values.reshape(ny, nx)


def plot_field_comparison(
    true_field: torch.Tensor | np.ndarray,
    pred_field: torch.Tensor | np.ndarray,
    resolution: int | tuple[int, int],
    obs_coords: torch.Tensor | np.ndarray | None = None,
    save_path: str | Path | None = None,
    title: str = "HNF Reconstruction",
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Plot true field, prediction, and absolute error side by side."""
    true_2d = _reshape_field(true_field, resolution)
    pred_2d = _reshape_field(pred_field, resolution)
    error_2d = np.abs(pred_2d - true_2d)

    vmax = np.max(np.abs(true_2d))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    panels = [
        (true_2d, "True Field"),
        (pred_2d, "Predicted Field"),
        (error_2d, "Absolute Error"),
    ]
    for ax, (data, label) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax if label != "Absolute Error" else None)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046)

        if obs_coords is not None and label == "True Field":
            oc = _to_numpy(obs_coords) if isinstance(obs_coords, torch.Tensor) else obs_coords
            if isinstance(resolution, int):
                ny = nx = resolution
            else:
                ny, nx = resolution
            xs = (oc[:, 0] + 1) / 2 * (nx - 1)
            ys = (oc[:, 1] + 1) / 2 * (ny - 1)
            ax.scatter(xs, ys, c="k", s=12, alpha=0.7, label="obs")

    fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reconstruction(
    sample,
    field_pred: torch.Tensor,
    save_path: str | Path | None = None,
    show_obs: bool = True,
) -> plt.Figure:
    """High-level wrapper: plot reconstruction from a SyntheticFieldSample."""
    title = f"HNF — {getattr(sample, 'field_type', 'field')}"
    return plot_field_comparison(
        true_field=sample.field_values,
        pred_field=field_pred,
        resolution=sample.resolution,
        obs_coords=sample.obs_coords if show_obs else None,
        save_path=save_path,
        title=title,
    )


def plot_observation_distribution(
    obs_coords: torch.Tensor,
    resolution: int | tuple[int, int] = 64,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of sparse observation locations on the unit square."""
    oc = _to_numpy(obs_coords)
    if isinstance(resolution, int):
        ny = nx = resolution
    else:
        ny, nx = resolution

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(oc[:, 0], oc[:, 1], c="crimson", s=20, alpha=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Observation Points (N={len(oc)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
