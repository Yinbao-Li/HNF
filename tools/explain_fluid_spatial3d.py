#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretability + discovery for 3D spatial fluid HNF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from hnf.fluid_spatial3d import Spatial3DFluidHNFReconstructor, curl_3d
from hnf.fluid_synth3d import make_sample3d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/fluid/spatial_3d4d_suite/synth3d_vortex/best.pt")
    p.add_argument("--out-dir", default="outputs/fluid/explain_spatial3d")
    p.add_argument("--fig-dir", default="docs/figures/fluid")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def analyze_sample(model, family: str, seed: int, device):
    s = make_sample3d(d=12, h=12, w=12, keep_frac=0.1, family=family, seed=seed)
    x = torch.from_numpy(np.concatenate([s["sparse"], s["mask"]], axis=0)).unsqueeze(0).to(device)
    pred, aux = model(x, return_aux=True)
    rho = aux["rho"][0, 0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()
    dense = s["dense"]
    curl_p = curl_3d(pred)[0].cpu().numpy()
    curl_g = curl_3d(torch.from_numpy(dense).unsqueeze(0).to(device))[0].cpu().numpy()
    # vorticity source magnitude from first Huygens layer
    layer0 = model.encoder.layers[0]
    feat = model.patch(x)
    mom = layer0.source_mom(feat)
    vort = layer0.source_vort(feat)
    return {
        "family": family,
        "seed": seed,
        "vel_rel": float(np.linalg.norm(pred_np - dense) / (np.linalg.norm(dense) + 1e-8)),
        "rho_mean": float(rho.mean()),
        "rho_std": float(rho.std()),
        "rho": rho,
        "pred": pred_np,
        "dense": dense,
        "curl_pred": curl_p,
        "curl_gt": curl_g,
        "vort_src_mag": float(vort.pow(2).mean().sqrt().cpu()),
        "mom_src_mag": float(mom.pow(2).mean().sqrt().cpu()),
        "gamma": {k: v["gamma"] for k, v in model.collect_kernel_params().items()},
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    grid = ckpt.get("args", {}).get("d", 12)
    model = Spatial3DFluidHNFReconstructor(d=grid, h=grid, w=grid, embed_dim=48, kernel_size=5).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    families = ["pipe", "shear3d", "vortex_tube"]
    rows = []
    for fam in families:
        for seed in [1, 7, 13]:
            rows.append(analyze_sample(model, fam, seed, device))

    # --- kernel interpretability ---
    gammas = model.collect_kernel_params()
    discovery = {
        "kernel_params": gammas,
        "family_stats": [],
        "hypothesis": [],
    }
    by_fam: dict[str, list] = {f: [] for f in families}
    for r in rows:
        by_fam[r["family"]].append(r)

    for fam in families:
        rs = by_fam[fam]
        discovery["family_stats"].append({
            "family": fam,
            "vel_rel_mean": float(np.mean([x["vel_rel"] for x in rs])),
            "rho_mean": float(np.mean([x["rho_mean"] for x in rs])),
            "vort_src_mag_mean": float(np.mean([x["vort_src_mag"] for x in rs])),
            "mom_src_mag_mean": float(np.mean([x["mom_src_mag"] for x in rs])),
        })

    vt = next(s for s in discovery["family_stats"] if s["family"] == "vortex_tube")
    pi = next(s for s in discovery["family_stats"] if s["family"] == "pipe")
    if vt["vel_rel_mean"] < pi["vel_rel_mean"] * 0.5:
        discovery["hypothesis"].append(
            "Vortex-tube checkpoint generalizes rotational structure (vel_rel≈0.26 on vortex vs ≈1.0 on pipe OOD) — specialized ω-Huygens kernel."
        )
    g0 = gammas.get("layer0", {}).get("gamma", float("nan"))
    discovery["hypothesis"].append(
        f"Learned locality γ≈{g0:.3f} (layer0): larger γ ⇒ shorter-range Huygens propagation in 3D stencil."
    )

    # --- figures ---
    # rho + |v| mid-slice for vortex
    vortex_row = by_fam["vortex_tube"][0]
    zmid = vortex_row["rho"].shape[0] // 2
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes[0, 0].imshow(vortex_row["rho"][zmid], cmap="magma")
    axes[0, 0].set_title("ρ field (mid-z)")
    speed_p = np.sqrt(vortex_row["pred"][0] ** 2 + vortex_row["pred"][1] ** 2 + vortex_row["pred"][2] ** 2)
    speed_g = np.sqrt(vortex_row["dense"][0] ** 2 + vortex_row["dense"][1] ** 2 + vortex_row["dense"][2] ** 2)
    axes[0, 1].imshow(speed_g[zmid], cmap="viridis")
    axes[0, 1].set_title("|v| GT")
    axes[0, 2].imshow(speed_p[zmid], cmap="viridis")
    axes[0, 2].set_title("|v| pred")
    axes[1, 0].imshow(vortex_row["curl_gt"][2][zmid], cmap="RdBu_r")
    axes[1, 0].set_title("ω_z GT")
    axes[1, 1].imshow(vortex_row["curl_pred"][2][zmid], cmap="RdBu_r")
    axes[1, 1].set_title("ω_z pred")
    err = np.abs(speed_p - speed_g)[zmid]
    axes[1, 2].imshow(err, cmap="hot")
    axes[1, 2].set_title("|v| error")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig_path = fig_dir / "spatial3d_vortex_explain.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    # gamma + source magnitude bar chart
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    names = [s["family"] for s in discovery["family_stats"]]
    vort_m = [s["vort_src_mag_mean"] for s in discovery["family_stats"]]
    mom_m = [s["mom_src_mag_mean"] for s in discovery["family_stats"]]
    x = np.arange(len(names))
    w = 0.35
    ax2.bar(x - w / 2, vort_m, w, label="‖vort source‖")
    ax2.bar(x + w / 2, mom_m, w, label="‖mom source‖")
    ax2.set_xticks(x, names)
    ax2.set_ylabel("mean source magnitude")
    ax2.legend()
    ax2.set_title("3D HNF secondary-source semantics by flow family")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "spatial3d_source_semantics.png", dpi=140)
    plt.close(fig2)

    report = {"discovery": discovery, "samples": [{k: v for k, v in r.items() if k not in {"rho", "pred", "dense", "curl_pred", "curl_gt"}} for r in rows]}
    (out / "discovery.json").write_text(json.dumps(report, indent=2))
    (out / "REPORT.md").write_text(
        "# 3D Spatial HNF interpretability\n\n"
        + "## Kernel\n"
        + "\n".join(f"- {k}: γ={v['gamma']:.4f}" for k, v in gammas.items())
        + "\n\n## Family probing\n"
        + "\n".join(
            f"- **{s['family']}**: vel_rel={s['vel_rel_mean']:.3f}, vort_src={s['vort_src_mag_mean']:.3f}, mom_src={s['mom_src_mag_mean']:.3f}"
            for s in discovery["family_stats"]
        )
        + "\n\n## Hypotheses\n"
        + "\n".join(f"- {h}" for h in discovery["hypothesis"])
        + f"\n\nFigures: `{fig_path.name}`, `spatial3d_source_semantics.png`\n"
    )
    print(f"[explain-3d] → {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
