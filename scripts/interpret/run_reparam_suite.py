#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5 reparameterization suite (README Part III.3).

Tracks:
  (1) Analytic medium: poly fits of rho summaries vs epicentral distance
  (2) Classical velocity residual: Physics Decoder vp/vs vs AK135 (Ambon table)
  (3) Operator simplification: SVD spectrum / low-rank approx error of K

Usage:
  PYTHONPATH=. python scripts/interpret/run_reparam_suite.py --device cuda
  PYTHONPATH=. python scripts/interpret/run_reparam_suite.py --compare ak135 --svd-ranks 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hnf.ambon_data import load_ambon_velocity_model
from hnf.inversion_1d import LayeredEarth1D, default_synth_model
from hnf.kernel import HuygensKernel
from hnf.physics_decoder import load_physics_decoder_from_checkpoint
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model as load_picking_ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HNF reparameterization suite (Step 5)")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--physics-head", default="outputs/physics_decoder_run28_macro/best_physics_head.pt")
    p.add_argument("--output-dir", default="outputs/reparam_suite_run28")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--n-events", type=int, default=128)
    p.add_argument("--compare", default="ak135", choices=["ak135", "synth", "both"])
    p.add_argument("--svd-ranks", default="1,2,4,8")
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def _finite(x: float) -> bool:
    return x is not None and math.isfinite(float(x))


def poly_fit_report(x: np.ndarray, y: np.ndarray, deg: int = 2) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < deg + 2:
        return {"n": int(x.size), "ok": False}
    coef = np.polyfit(x, y, deg)
    yhat = np.polyval(coef, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {
        "n": int(x.size),
        "ok": True,
        "degree": deg,
        "coef_high_to_low": [float(c) for c in coef],
        "r2": float(r2),
        "rmse": rmse,
    }


def collect_event_rows(
    backbone: torch.nn.Module,
    decoder,
    device: torch.device,
    seq_len: int,
    n_events: int,
    seed: int,
) -> list[dict]:
    ds = STEADPickingDataset("test", seq_len=seq_len, max_event_traces=max(400, n_events * 3), seed=seed)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    rows: list[dict] = []
    kernel_params = backbone.collect_kernel_params()
    for batch in loader:
        if len(rows) >= n_events:
            break
        if float(batch["det"][0]) <= 0.5:
            continue
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        if not _finite(dist):
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        with torch.no_grad():
            feat = decoder.extract_station_features(x, t, include_picks=True)
            out, _ = decoder.forward_event(x, t, include_picks=True)
        rho = feat["rho"][0].detach().cpu().numpy()
        vp = out.vp[0].detach().cpu().numpy()
        vs = out.vs[0].detach().cpu().numpy()
        rows.append(
            {
                "distance_km": dist,
                "depth_km": depth if _finite(depth) else float("nan"),
                "rho_mean": float(np.mean(rho)),
                "rho_peak": float(np.max(rho)),
                "gamma_p0": float(kernel_params.get("p_branch_0", {}).get("gamma", float("nan"))),
                "omega_p0": float(kernel_params.get("p_branch_0", {}).get("omega", float("nan"))),
                "kernel_vp": float(feat["kernel_vp"][0]),
                "kernel_vs": float(feat["kernel_vs"][0]),
                "vp": [float(v) for v in vp],
                "vs": [float(v) for v in vs],
                "vp_mean": float(np.mean(vp)),
                "vs_mean": float(np.mean(vs)),
                "vpvs_mean": float(np.mean(vp / np.clip(vs, 1e-6, None))),
            }
        )
    return rows


def section_analytic_medium(rows: list[dict], out_dir: Path) -> dict:
    dist = np.array([r["distance_km"] for r in rows], dtype=np.float64)
    fits = {
        "rho_mean_vs_distance": poly_fit_report(dist, np.array([r["rho_mean"] for r in rows])),
        "rho_peak_vs_distance": poly_fit_report(dist, np.array([r["rho_peak"] for r in rows])),
        "vp_mean_vs_distance": poly_fit_report(dist, np.array([r["vp_mean"] for r in rows])),
        "vpvs_mean_vs_distance": poly_fit_report(dist, np.array([r["vpvs_mean"] for r in rows])),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    panels = [
        (axes[0, 0], "rho_mean", "rho_mean_vs_distance", "C0"),
        (axes[0, 1], "rho_peak", "rho_peak_vs_distance", "C1"),
        (axes[1, 0], "vp_mean", "vp_mean_vs_distance", "C2"),
        (axes[1, 1], "vpvs_mean", "vpvs_mean_vs_distance", "C3"),
    ]
    xs = np.linspace(float(np.nanmin(dist)), float(np.nanmax(dist)), 200)
    for ax, key, fit_key, color in panels:
        ys = np.array([r[key] for r in rows], dtype=np.float64)
        ax.scatter(dist, ys, s=28, alpha=0.75, c=color)
        fit = fits[fit_key]
        if fit.get("ok"):
            ax.plot(xs, np.polyval(fit["coef_high_to_low"], xs), "k-", lw=2, label=f"poly R²={fit['r2']:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("epicentral distance (km)")
        ax.set_ylabel(key)
        ax.set_title(fit_key.replace("_", " "))
        ax.grid(True, alpha=0.3)
    fig_path = out_dir / "analytic_medium_distance_fits.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return {"figure": str(fig_path), "fits": fits, "n": len(rows)}


def _interp_profile(depths_src: np.ndarray, values: np.ndarray, z_query: np.ndarray) -> np.ndarray:
    """Piecewise-constant layer values on interface depths -> sample at midpoints z_query."""
    # depths_src: (L+1,), values: (L,)
    mids = 0.5 * (depths_src[:-1] + depths_src[1:])
    return np.interp(z_query, mids, values, left=values[0], right=values[-1])


def section_velocity_residual(
    rows: list[dict],
    out_dir: Path,
    compare: str,
    decoder_depths: torch.Tensor,
) -> dict:
    refs = {}
    if compare in {"ak135", "both"}:
        try:
            refs["ak135"] = load_ambon_velocity_model(use_velest=False)
        except Exception as exc:  # noqa: BLE001
            refs["ak135_error"] = str(exc)
    if compare in {"synth", "both"} or "ak135" not in refs:
        refs["synth"] = default_synth_model("cpu")

    # Decoder outputs are on its own layer grid (usually synth depths).
    d_dec = decoder_depths.detach().cpu().numpy()
    z_mid = 0.5 * (d_dec[:-1] + d_dec[1:])

    report: dict = {"decoder_depths_km": d_dec.tolist(), "references": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    for name, earth in list(refs.items()):
        if not isinstance(earth, LayeredEarth1D):
            continue
        d_ref = earth.depths.detach().cpu().numpy()
        vp_ref = _interp_profile(d_ref, earth.vp.detach().cpu().numpy(), z_mid)
        vs_ref = _interp_profile(d_ref, earth.vs.detach().cpu().numpy(), z_mid)
        vp_pred = np.array([r["vp"] for r in rows], dtype=np.float64)
        vs_pred = np.array([r["vs"] for r in rows], dtype=np.float64)
        # Broadcast ref across events
        dvp = vp_pred - vp_ref[None, :]
        dvs = vs_pred - vs_ref[None, :]
        mae_vp = float(np.mean(np.abs(dvp)))
        mae_vs = float(np.mean(np.abs(dvs)))
        bias_vp = float(np.mean(dvp))
        bias_vs = float(np.mean(dvs))
        report["references"][name] = {
            "mae_vp": mae_vp,
            "mae_vs": mae_vs,
            "bias_vp": bias_vp,
            "bias_vs": bias_vs,
            "layer_mae_vp": [float(x) for x in np.mean(np.abs(dvp), axis=0)],
            "layer_mae_vs": [float(x) for x in np.mean(np.abs(dvs), axis=0)],
            "ref_depths_km": d_ref.tolist(),
            "ref_vp": earth.vp.detach().cpu().tolist(),
            "ref_vs": earth.vs.detach().cpu().tolist(),
        }

        axes[0].plot(z_mid, vp_ref, "k--", lw=2, label=f"{name} ref")
        axes[1].plot(z_mid, vs_ref, "k--", lw=2, label=f"{name} ref")
        axes[0].plot(z_mid, vp_pred.mean(0), "C0-", lw=2, label="decoder mean")
        axes[0].fill_between(
            z_mid,
            vp_pred.mean(0) - vp_pred.std(0),
            vp_pred.mean(0) + vp_pred.std(0),
            color="C0",
            alpha=0.2,
        )
        axes[1].plot(z_mid, vs_pred.mean(0), "C1-", lw=2, label="decoder mean")
        axes[1].fill_between(
            z_mid,
            vs_pred.mean(0) - vs_pred.std(0),
            vs_pred.mean(0) + vs_pred.std(0),
            color="C1",
            alpha=0.2,
        )
        # Plot the primary classical reference only (ak135 preferred).
        break

    axes[0].set_xlabel("depth (km)")
    axes[0].set_ylabel("Vp (km/s)")
    axes[0].set_title("Decoder Vp vs classical reference")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("depth (km)")
    axes[1].set_ylabel("Vs (km/s)")
    axes[1].set_title("Decoder Vs vs classical reference")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig_path = out_dir / "velocity_residual_vs_classical.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    report["figure"] = str(fig_path)
    if "ak135_error" in refs:
        report["ak135_error"] = refs["ak135_error"]
    return report


def section_kernel_svd(backbone: torch.nn.Module, out_dir: Path, ranks: list[int], device: torch.device) -> dict:
    params = backbone.collect_kernel_params()
    # Prefer P-branch Fresnel kernel if present.
    key = "p_branch_0" if "p_branch_0" in params else next(iter(params))
    g = float(params[key]["gamma"])
    w = float(params[key]["omega"])
    c = float(params[key]["wave_speed"])

    n = 160
    t = torch.linspace(0, 15.0, n, device=device).view(1, n, 1)
    x = torch.zeros(1, n, 4, device=device)
    k = HuygensKernel(
        gamma=g,
        omega=w,
        causal=True,
        wave_speed=c,
        distance_mode="time",
        local_window_sec=15.0,
        principle="huygens_fresnel",
        obliquity_scale=1.0,
    ).to(device)
    with torch.no_grad():
        K = torch.abs(k(x, t=t, return_complex=True))[0].detach().cpu().numpy()
    # SVD on real magnitude kernel
    u, s, vh = np.linalg.svd(K, full_matrices=False)
    energy = (s**2) / (np.sum(s**2) + 1e-12)
    cum = np.cumsum(energy)

    recon_err = {}
    for r in ranks:
        rr = max(1, min(int(r), len(s)))
        Kr = (u[:, :rr] * s[:rr]) @ vh[:rr, :]
        recon_err[str(rr)] = {
            "rel_fro": float(np.linalg.norm(K - Kr) / (np.linalg.norm(K) + 1e-12)),
            "cum_energy": float(cum[rr - 1]),
        }

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    im = axes[0].imshow(K, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"|K| ({key})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].semilogy(np.arange(1, len(s) + 1), s, "C0-")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("singular value")
    axes[1].set_title("SVD spectrum")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(np.arange(1, len(cum) + 1), cum, "C2-")
    for r, info in recon_err.items():
        axes[2].scatter([int(r)], [info["cum_energy"]], s=40, label=f"r={r} err={info['rel_fro']:.3f}")
    axes[2].set_xlabel("rank")
    axes[2].set_ylabel("cumulative energy")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Low-rank capture")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)
    fig_path = out_dir / "kernel_svd_lowrank.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "figure": str(fig_path),
        "kernel_key": key,
        "gamma": g,
        "omega": w,
        "wave_speed": c,
        "top10_singular": [float(x) for x in s[:10]],
        "cum_energy_at_ranks": {str(r): float(cum[min(int(r), len(cum)) - 1]) for r in ranks},
        "recon_rel_fro": recon_err,
    }


def write_report(out_dir: Path, payload: dict) -> None:
    md = out_dir / "reparam_report.md"
    analytic = payload["analytic_medium"]["fits"]
    lines = [
        "# Reparameterization suite (Step 5)",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- physics head: `{payload['physics_head']}`",
        f"- n events: {payload['n_events']}",
        "",
        "## (1) Analytic medium vs distance",
        "",
    ]
    for name, fit in analytic.items():
        if fit.get("ok"):
            lines.append(
                f"- `{name}`: R²={fit['r2']:.3f}, RMSE={fit['rmse']:.4f}, coef={fit['coef_high_to_low']}"
            )
        else:
            lines.append(f"- `{name}`: insufficient points")
    lines += ["", f"Figure: `{payload['analytic_medium']['figure']}`", "", "## (2) Velocity residual vs classical", ""]
    for ref, stats in payload["velocity_residual"].get("references", {}).items():
        lines.append(
            f"- `{ref}`: MAE Vp={stats['mae_vp']:.3f}, Vs={stats['mae_vs']:.3f}; "
            f"bias Vp={stats['bias_vp']:.3f}, Vs={stats['bias_vs']:.3f}"
        )
    if "ak135_error" in payload["velocity_residual"]:
        lines.append(f"- AK135 load failed: {payload['velocity_residual']['ak135_error']}")
    lines += ["", f"Figure: `{payload['velocity_residual']['figure']}`", "", "## (3) Kernel low-rank SVD", ""]
    svd = payload["kernel_svd"]
    lines.append(f"- kernel `{svd['kernel_key']}` γ={svd['gamma']:.3f} ω={svd['omega']:.3f} c={svd['wave_speed']:.3f}")
    for r, info in svd["recon_rel_fro"].items():
        lines.append(f"- rank {r}: cum_energy={info['cum_energy']:.3f}, rel_fro_err={info['rel_fro']:.3f}")
    lines += ["", f"Figure: `{svd['figure']}`", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "reparam_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ranks = [int(x) for x in args.svd_ranks.split(",") if x.strip()]

    backbone, ckpt_args = load_picking_ckpt(Path(args.checkpoint), device, bypass_noise_cancel=True)
    # Multi-scale / fresnel flags may live in ckpt; load_model already rebuilds.
    decoder = load_physics_decoder_from_checkpoint(
        backbone=backbone,
        physics_head_path=args.physics_head,
        device=device,
        head_mode="macro",
        embed_dim=int(ckpt_args.get("embed_dim", 64)),
        infer_seq_len=min(600, int(args.seq_len)),
    )

    rows = collect_event_rows(
        backbone=backbone,
        decoder=decoder,
        device=device,
        seq_len=args.seq_len,
        n_events=args.n_events,
        seed=args.seed,
    )
    if not rows:
        raise RuntimeError("No valid STEAD events collected for reparam suite")

    analytic = section_analytic_medium(rows, out_dir)
    # Physics head uses default_synth_model depths for macro layers.
    vel = section_velocity_residual(
        rows,
        out_dir,
        compare=args.compare,
        decoder_depths=default_synth_model("cpu").depths,
    )
    svd = section_kernel_svd(backbone, out_dir, ranks=ranks, device=device)

    # Lightweight row export (drop long vectors for CSV-ish JSON)
    slim_rows = [
        {k: v for k, v in r.items() if k not in {"vp", "vs"}}
        for r in rows
    ]
    payload = {
        "checkpoint": args.checkpoint,
        "physics_head": args.physics_head,
        "n_events": len(rows),
        "compare": args.compare,
        "svd_ranks": ranks,
        "analytic_medium": analytic,
        "velocity_residual": vel,
        "kernel_svd": svd,
        "rows": slim_rows,
    }
    write_report(out_dir, payload)
    print(json.dumps({"output_dir": str(out_dir), "n_events": len(rows), "report": str(out_dir / "reparam_report.md")}, indent=2))


if __name__ == "__main__":
    main()
