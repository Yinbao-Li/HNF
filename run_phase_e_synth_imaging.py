#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase E: synthetic imaging closed loop.

Build a quasi-2D truth model from laterally varying 1D columns, recover local
1D models with existing inversion code, then assemble 2D sections for:
  - true / recovered Vp, Vs
  - error maps
  - ray-hit coverage
  - bootstrap uncertainty

This is the first "geologic image output" loop intended for README figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hnf.inv_plot import perturb_initial, plot_velocity_profiles
from hnf.inversion_1d import (
    LayeredEarth1D,
    default_synth_model,
    invert_layered_1d,
    model_rmse,
    synthesize_travel_times,
)
from hnf.ray_paths import direct_ray_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase E synthetic 2D imaging closed loop")
    p.add_argument("--output-dir", default="outputs/phase_e_synth_imaging")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-type", choices=["smooth_anomaly", "marmousi_style"], default="marmousi_style")
    p.add_argument("--n-columns", type=int, default=9)
    p.add_argument("--x-max", type=float, default=50.0)
    p.add_argument("--z-max", type=float, default=35.0)
    p.add_argument("--nx-grid", type=int, default=121)
    p.add_argument("--nz-grid", type=int, default=141)
    p.add_argument("--source-depth", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--n-boot", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-std", type=float, default=0.015)
    return p.parse_args()


def _gaussian(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2)


def build_quasi_2d_truth(
    x_columns: torch.Tensor,
    base: LayeredEarth1D,
    model_type: str = "smooth_anomaly",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Laterally varying 1D columns with two smooth anomalies.

    Returns layerwise Vp/Vs arrays of shape (n_columns, n_layers).
    """
    centers = 0.5 * (base.depths[:-1] + base.depths[1:])
    x_norm = x_columns / x_columns.max().clamp(min=1e-6)

    if model_type == "marmousi_style":
        left_low = -0.55 * _gaussian(x_norm, 0.18, 0.09)
        mid_ridge = 0.45 * _gaussian(x_norm, 0.48, 0.11)
        right_high = 0.38 * _gaussian(x_norm, 0.78, 0.08)
        deep_trend = 0.22 * torch.sin(2.0 * torch.pi * (x_norm - 0.15))
    else:
        left_low = -0.45 * _gaussian(x_norm, 0.28, 0.12)
        mid_ridge = torch.zeros_like(x_norm)
        right_high = 0.35 * _gaussian(x_norm, 0.73, 0.10)
        deep_trend = 0.18 * (x_norm - 0.5)

    vp_cols = []
    vs_cols = []
    for i, zc in enumerate(centers):
        if model_type == "marmousi_style":
            depth_low = _gaussian(zc.view(1), 7.0, 3.2)[0]
            depth_mid = _gaussian(zc.view(1), 14.0, 4.0)[0]
            depth_high = _gaussian(zc.view(1), 23.0, 5.8)[0]
            wedge = (zc / centers.max()) ** 1.4
            layer_scale = (
                left_low * depth_low
                + mid_ridge * depth_mid
                + right_high * depth_high
                + deep_trend * wedge
            )
        else:
            depth_low = _gaussian(zc.view(1), 8.0, 4.0)[0]
            depth_high = _gaussian(zc.view(1), 18.0, 5.5)[0]
            layer_scale = left_low * depth_low + right_high * depth_high + deep_trend * (zc / centers.max())

        vp = base.vp[i] * (1.0 + 0.16 * layer_scale)
        vs = base.vs[i] * (1.0 + 0.13 * layer_scale)
        vp_cols.append(vp)
        vs_cols.append(torch.minimum(vs, vp * 0.75))

    vp = torch.stack(vp_cols, dim=1).clamp(min=1.5)
    vs = torch.stack(vs_cols, dim=1).clamp(min=1.0)

    vp_out = []
    vs_out = []
    for col in range(vp.shape[0]):
        vpc = vp[col].clone()
        vsc = vs[col].clone()
        for i in range(1, vpc.numel()):
            vpc[i] = torch.maximum(vpc[i], vpc[i - 1] + 0.05)
            vsc[i] = torch.minimum(vsc[i], vpc[i] * 0.75)
        vp_out.append(vpc)
        vs_out.append(vsc)
    return torch.stack(vp_out), torch.stack(vs_out)


def build_column_earth(
    base: LayeredEarth1D,
    vp: torch.Tensor,
    vs: torch.Tensor,
) -> LayeredEarth1D:
    return LayeredEarth1D(depths=base.depths, vp=vp, vs=vs, q=base.q)


def interpolate_layers_to_section(
    x_columns: torch.Tensor,
    layer_values: torch.Tensor,
    depths: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Interpolate laterally in each layer, keep piecewise-constant vertical structure."""
    vals_np = layer_values.detach().cpu().numpy()
    x_cols_np = x_columns.detach().cpu().numpy()
    xg_np = x_grid.detach().cpu().numpy()
    layer_interp = []
    for i in range(layer_values.shape[1]):
        layer_interp.append(np.interp(xg_np, x_cols_np, vals_np[:, i]))
    layer_interp = np.stack(layer_interp, axis=0)  # (n_layers, nx)

    section = np.zeros((len(zg_np := z_grid.detach().cpu().numpy()), len(xg_np)), dtype=np.float32)
    depths_np = depths.detach().cpu().numpy()
    for iz, z in enumerate(zg_np):
        layer_idx = np.searchsorted(depths_np, z, side="right") - 1
        layer_idx = int(np.clip(layer_idx, 0, layer_values.shape[1] - 1))
        section[iz] = layer_interp[layer_idx]
    return torch.from_numpy(section)


def rasterize_paths(
    earth: LayeredEarth1D,
    x0: float,
    offsets: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    hit = torch.zeros(z_grid.numel(), x_grid.numel(), dtype=torch.float32)
    dx = float(x_grid[1] - x_grid[0]) if x_grid.numel() > 1 else 1.0
    dz = float(z_grid[1] - z_grid[0]) if z_grid.numel() > 1 else 1.0

    for off in offsets.detach().cpu().tolist():
        if abs(off) < 1e-8:
            continue
        for sign in (-1.0, 1.0):
            path_x, path_z = direct_ray_path(earth, "P", float(min(earth.depths[-1] - 1e-3, 10.0)), float(abs(off)))
            px = sign * path_x + x0
            pz = path_z
            if (px.max() < float(x_grid.min())) or (px.min() > float(x_grid.max())):
                continue
            for i in range(len(px) - 1):
                seg_len = max(
                    int(
                        torch.ceil(
                            torch.maximum(
                                (px[i + 1] - px[i]).abs() / dx,
                                (pz[i + 1] - pz[i]).abs() / dz,
                            )
                        ).item()
                    ),
                    1,
                )
                xs = torch.linspace(px[i], px[i + 1], seg_len + 1)
                zs = torch.linspace(pz[i], pz[i + 1], seg_len + 1)
                ix = torch.bucketize(xs, x_grid).clamp(1, x_grid.numel()) - 1
                iz = torch.bucketize(zs, z_grid).clamp(1, z_grid.numel()) - 1
                hit[iz, ix] += 1.0
    return hit


def plot_section(
    section: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
    out_path: Path,
    *,
    title: str,
    cmap: str = "viridis",
    cbar_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    x_columns: torch.Tensor | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(
        section.detach().cpu().numpy(),
        extent=[float(x_grid[0]), float(x_grid[-1]), float(z_grid[-1]), float(z_grid[0])],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if x_columns is not None:
        for x in x_columns.detach().cpu().tolist():
            ax.axvline(x, color="white", alpha=0.18, lw=0.8)
    ax.set_xlabel("Profile distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.95)
    cb.set_label(cbar_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_summary_panel(
    out_path: Path,
    *,
    true_vp: torch.Tensor,
    rec_vp: torch.Tensor,
    err_vp: torch.Tensor,
    coverage: torch.Tensor,
    unc_vp: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    items = [
        (true_vp, "True Vp", "viridis"),
        (rec_vp, "Recovered Vp", "viridis"),
        (err_vp, "Vp Error", "coolwarm"),
        (coverage, "Ray Coverage", "magma"),
        (unc_vp, "Vp Uncertainty", "plasma"),
    ]
    for ax, (section, title, cmap) in zip(axes.flat, items):
        im = ax.imshow(
            section.detach().cpu().numpy(),
            extent=[float(x_grid[0]), float(x_grid[-1]), float(z_grid[-1]), float(z_grid[0])],
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Depth (km)")
        fig.colorbar(im, ax=ax, shrink=0.86)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.02,
        0.95,
        "Synthetic Closed Loop\n\n"
        "1. Known quasi-2D truth\n"
        "2. Local 1D recoveries\n"
        "3. Assembled 2D section\n"
        "4. Coverage + uncertainty",
        va="top",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_geometry(
    x_columns: torch.Tensor,
    offsets: torch.Tensor,
    source_depth: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    for x0 in x_columns.detach().cpu().tolist():
        ax.scatter([x0], [source_depth], c="C3", s=18)
        for off in offsets.detach().cpu().tolist():
            if abs(off) < 1e-8:
                ax.scatter([x0], [0.0], c="black", s=8)
                continue
            ax.scatter([x0 - off, x0 + off], [0.0, 0.0], c="C0", s=8, alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Profile distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Synthetic acquisition geometry")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = default_synth_model(device)
    x_columns = torch.linspace(0.0, args.x_max, args.n_columns, device=device)
    x_grid = torch.linspace(0.0, args.x_max, args.nx_grid, device=device)
    z_grid = torch.linspace(0.0, args.z_max, args.nz_grid, device=device)
    offsets = torch.tensor([0.0, 4.0, 8.0, 12.0, 16.0], device=device)

    true_vp_cols, true_vs_cols = build_quasi_2d_truth(x_columns, base, model_type=args.model_type)
    rec_vp_cols = []
    rec_vs_cols = []
    unc_vp_cols = []
    unc_vs_cols = []
    rmse_rows = []
    coverage = torch.zeros(z_grid.numel(), x_grid.numel(), dtype=torch.float32)

    for col_idx, x0 in enumerate(x_columns.detach().cpu().tolist()):
        true_earth = build_column_earth(base, true_vp_cols[col_idx], true_vs_cols[col_idx])
        coverage += rasterize_paths(true_earth, x0, offsets[1:], x_grid.cpu(), z_grid.cpu())
        obs = synthesize_travel_times(
            true_earth,
            args.source_depth,
            offsets,
            noise_std=args.noise_std,
            seed=args.seed + col_idx * 17,
        )

        vp_ens = []
        vs_ens = []
        histories = []
        for boot in range(args.n_boot):
            vp_init, vs_init, q_init = perturb_initial(
                true_earth.vp, true_earth.vs, true_earth.q,
                seed=args.seed + col_idx * 101 + boot * 13,
                q_scale=1.0,
            )
            inv_model, hist = invert_layered_1d(
                true_earth.depths,
                vp_init,
                vs_init,
                q_init,
                args.source_depth,
                offsets,
                obs,
                steps=args.steps,
                lr=0.055,
                smooth_weight=0.04,
                anchor_weight=0.005,
                verbose=False,
            )
            vp_ens.append(inv_model.vp.detach())
            vs_ens.append(inv_model.vs.detach())
            histories.append(hist)

            if boot == 0 and col_idx in {0, len(x_columns) // 2, len(x_columns) - 1}:
                plot_velocity_profiles(
                    true_earth.depths,
                    true_earth.vp,
                    true_earth.vs,
                    vp_init,
                    vs_init,
                    inv_model.vp.detach(),
                    inv_model.vs.detach(),
                    fig_dir / f"local_profile_col{col_idx:02d}.png",
                    title=f"Local 1D inversion at x={x0:.1f} km",
                )

        vp_ens_t = torch.stack(vp_ens)
        vs_ens_t = torch.stack(vs_ens)
        rec_vp = vp_ens_t.mean(dim=0)
        rec_vs = vs_ens_t.mean(dim=0)
        unc_vp = vp_ens_t.std(dim=0)
        unc_vs = vs_ens_t.std(dim=0)
        rec_vp_cols.append(rec_vp)
        rec_vs_cols.append(rec_vs)
        unc_vp_cols.append(unc_vp)
        unc_vs_cols.append(unc_vs)

        rmse = model_rmse(
            true_earth,
            LayeredEarth1D(true_earth.depths, rec_vp, rec_vs, true_earth.q),
        )
        rmse_rows.append(
            {
                "x_km": x0,
                "vp_rmse": rmse["vp_rmse"],
                "vs_rmse": rmse["vs_rmse"],
                "final_loss": histories[0][-1]["loss"] if histories[0] else None,
            }
        )
        print(
            f"[phase-e {col_idx + 1}/{len(x_columns)}] x={x0:.1f} "
            f"vp_rmse={rmse['vp_rmse']:.3f} vs_rmse={rmse['vs_rmse']:.3f}",
            flush=True,
        )

    rec_vp_cols_t = torch.stack(rec_vp_cols)
    rec_vs_cols_t = torch.stack(rec_vs_cols)
    unc_vp_cols_t = torch.stack(unc_vp_cols)
    unc_vs_cols_t = torch.stack(unc_vs_cols)

    true_vp_sec = interpolate_layers_to_section(x_columns, true_vp_cols, base.depths, x_grid, z_grid)
    true_vs_sec = interpolate_layers_to_section(x_columns, true_vs_cols, base.depths, x_grid, z_grid)
    rec_vp_sec = interpolate_layers_to_section(x_columns, rec_vp_cols_t, base.depths, x_grid, z_grid)
    rec_vs_sec = interpolate_layers_to_section(x_columns, rec_vs_cols_t, base.depths, x_grid, z_grid)
    unc_vp_sec = interpolate_layers_to_section(x_columns, unc_vp_cols_t, base.depths, x_grid, z_grid)
    unc_vs_sec = interpolate_layers_to_section(x_columns, unc_vs_cols_t, base.depths, x_grid, z_grid)
    err_vp_sec = rec_vp_sec - true_vp_sec
    err_vs_sec = rec_vs_sec - true_vs_sec

    plot_geometry(x_columns.cpu(), offsets.cpu(), args.source_depth, fig_dir / "acquisition_geometry.png")
    plot_section(true_vp_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "true_vp_2d.png", title="True Vp section", cbar_label="Vp (km/s)", x_columns=x_columns.cpu())
    plot_section(rec_vp_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "recovered_vp_2d.png", title="Recovered Vp section", cbar_label="Vp (km/s)", x_columns=x_columns.cpu())
    plot_section(err_vp_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "vp_error_2d.png", title="Vp error (recovered - true)", cmap="coolwarm", cbar_label="Vp error (km/s)", x_columns=x_columns.cpu())
    plot_section(true_vs_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "true_vs_2d.png", title="True Vs section", cbar_label="Vs (km/s)", x_columns=x_columns.cpu())
    plot_section(rec_vs_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "recovered_vs_2d.png", title="Recovered Vs section", cbar_label="Vs (km/s)", x_columns=x_columns.cpu())
    plot_section(err_vs_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "vs_error_2d.png", title="Vs error (recovered - true)", cmap="coolwarm", cbar_label="Vs error (km/s)", x_columns=x_columns.cpu())
    plot_section(coverage, x_grid.cpu(), z_grid.cpu(), fig_dir / "ray_coverage_2d.png", title="Ray-hit coverage", cmap="magma", cbar_label="Hit count", x_columns=x_columns.cpu())
    plot_section(unc_vp_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "vp_uncertainty_2d.png", title="Bootstrap Vp uncertainty", cmap="plasma", cbar_label="Vp std (km/s)", x_columns=x_columns.cpu())
    plot_section(unc_vs_sec, x_grid.cpu(), z_grid.cpu(), fig_dir / "vs_uncertainty_2d.png", title="Bootstrap Vs uncertainty", cmap="plasma", cbar_label="Vs std (km/s)", x_columns=x_columns.cpu())
    plot_summary_panel(
        fig_dir / "phase_e_summary_panel.png",
        true_vp=true_vp_sec,
        rec_vp=rec_vp_sec,
        err_vp=err_vp_sec,
        coverage=coverage,
        unc_vp=unc_vp_sec,
        x_grid=x_grid.cpu(),
        z_grid=z_grid.cpu(),
    )

    report = {
        "phase": "E_synth_imaging",
        "model_type": args.model_type,
        "n_columns": args.n_columns,
        "n_boot": args.n_boot,
        "steps": args.steps,
        "noise_std": args.noise_std,
        "mean_vp_rmse": float(np.mean([r["vp_rmse"] for r in rmse_rows])),
        "mean_vs_rmse": float(np.mean([r["vs_rmse"] for r in rmse_rows])),
        "max_vp_uncertainty": float(unc_vp_sec.max()),
        "max_vs_uncertainty": float(unc_vs_sec.max()),
        "coverage_nonzero_frac": float((coverage > 0).float().mean()),
        "figures": {
            "geometry": str(fig_dir / "acquisition_geometry.png"),
            "true_vp": str(fig_dir / "true_vp_2d.png"),
            "recovered_vp": str(fig_dir / "recovered_vp_2d.png"),
            "vp_error": str(fig_dir / "vp_error_2d.png"),
            "coverage": str(fig_dir / "ray_coverage_2d.png"),
            "vp_uncertainty": str(fig_dir / "vp_uncertainty_2d.png"),
            "summary_panel": str(fig_dir / "phase_e_summary_panel.png"),
        },
        "per_column": rmse_rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[phase-e] -> {out_dir}")


if __name__ == "__main__":
    main()
