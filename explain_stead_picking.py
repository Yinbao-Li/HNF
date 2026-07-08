#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize interpretable Huygens picking inference on STEAD samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hnf.picking_model import build_picking_model, load_picking_model_state
from hnf.stead_picking_dataset import STEADPickingDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain HNF picking inference")
    p.add_argument("--checkpoint", default="outputs/stead_hnf_picking_run7/best.pt")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=400)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = build_picking_model(
        embed_dim=ckpt_args.get("embed_dim", 64),
        num_shared_layers=ckpt_args.get("num_shared_layers", 2),
        num_branch_layers=ckpt_args.get("num_branch_layers", 2),
        gamma=ckpt_args.get("gamma", 0.5),
        omega=ckpt_args.get("omega", 0.3),
        vp=ckpt_args.get("vp", 8.0),
        vs=ckpt_args.get("vs", 4.5),
        local_window_sec=ckpt_args.get("local_window_sec", 15.0),
        dropout=ckpt_args.get("dropout", 0.1),
        per_time_det=ckpt_args.get("per_time_det", False),
        pick_head_hidden=ckpt_args.get("pick_head_hidden", 24),
        pick_head_kernel=ckpt_args.get("pick_head_kernel", 7),
        pick_head_layers=ckpt_args.get("pick_head_layers", 3),
        multi_scale=ckpt_args.get("multi_scale", False),
        residual_pick_head=ckpt_args.get("residual_pick_head", True),
        residual_det_head=ckpt_args.get("residual_det_head", True),
    ).to(device)
    load_picking_model_state(model, ckpt["state_dict"], strict=False)
    model.eval()
    return model


def pick_idx(prob: np.ndarray, threshold: float) -> int:
    idx = int(prob.argmax())
    return idx if prob[idx] >= threshold else -1


def plot_example(
    out_path: Path,
    sample: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    kernel_params: dict[str, dict[str, float]],
    pick_threshold: float,
    title: str,
) -> dict:
    x = sample["x"][0].cpu().numpy()
    t_sec = sample["t"][0, :, 0].cpu().numpy()
    p_prob = torch.sigmoid(outputs["p"][0]).cpu().numpy()
    s_prob = torch.sigmoid(outputs["s"][0]).cpu().numpy()
    rho = outputs["rho"][0].cpu().numpy()
    p_env = outputs["p_envelope"][0].cpu().numpy()
    s_env = outputs["s_envelope"][0].cpu().numpy()
    wave_e = outputs["wave_energy"][0].cpu().numpy()

    p_pred = pick_idx(p_prob, pick_threshold)
    s_pred = pick_idx(s_prob, pick_threshold)
    p_true = int(sample["p_idx"][0].item()) if sample["p_valid"][0] > 0 else -1
    s_true = int(sample["s_idx"][0].item()) if sample["s_valid"][0] > 0 else -1

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    comp_names = ["E", "N", "Z"]
    for i in range(3):
        axes[0].plot(t_sec, x[:, i], lw=0.7, alpha=0.85, label=comp_names[i])
    axes[0].set_ylabel("Waveform")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_title(title)

    axes[1].plot(t_sec, rho, color="saddlebrown", lw=1.0)
    axes[1].set_ylabel("rho(t)")
    axes[1].set_title("Medium density (noise/attenuation proxy)")

    det_logit = float(outputs["det"][0]) if outputs["det"].ndim == 0 else float(outputs["det"][0].item())
    det_prob = 1.0 / (1.0 + np.exp(-det_logit))

    axes[2].plot(t_sec, wave_e, color="gray", lw=0.9, label="shared energy")
    axes[2].axhline(det_prob, color="black", ls="--", lw=1.0, label=f"det prob={det_prob:.2f}")
    axes[2].set_ylabel("Detection")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].plot(t_sec, p_prob, color="C0", lw=1.2, label="P pick prob")
    if p_true >= 0:
        axes[3].axvline(t_sec[p_true], color="green", ls="--", lw=1.0, label="P true")
    if p_pred >= 0:
        axes[3].axvline(t_sec[p_pred], color="red", ls=":", lw=1.2, label="P pred")
    axes[3].set_ylabel("P branch")
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].plot(t_sec, s_env, color="C1", lw=0.8, alpha=0.6, label="S envelope")
    axes[4].plot(t_sec, s_prob, color="C1", lw=1.2, label="S pick prob")
    if s_true >= 0:
        axes[4].axvline(t_sec[s_true], color="green", ls="--", lw=1.0, label="S true")
    if s_pred >= 0:
        axes[4].axvline(t_sec[s_pred], color="red", ls=":", lw=1.2, label="S pred")
    axes[4].set_ylabel("S branch")
    axes[4].legend(loc="upper right", fontsize=8)

    if "kernel_contrib" in outputs and p_pred >= 0:
        contrib = outputs["kernel_contrib"][0].cpu().numpy()
        axes[5].plot(t_sec, contrib, color="purple", lw=0.9)
        axes[5].axvline(t_sec[p_pred], color="red", ls=":", lw=1.0)
        axes[5].set_ylabel("|K[p,:]|")
        axes[5].set_title("Huygens causal contributions to P pick time")
    else:
        axes[5].text(0.5, 0.5, "No kernel row (no P pick)", ha="center", va="center")
        axes[5].set_ylabel("Kernel")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "p_true_idx": p_true,
        "s_true_idx": s_true,
        "p_pred_idx": p_pred,
        "s_pred_idx": s_pred,
        "p_true_sec": float(t_sec[p_true]) if p_true >= 0 else None,
        "s_true_sec": float(t_sec[s_true]) if s_true >= 0 else None,
        "p_pred_sec": float(t_sec[p_pred]) if p_pred >= 0 else None,
        "s_pred_sec": float(t_sec[s_pred]) if s_pred >= 0 else None,
        "kernel_params": kernel_params,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(ckpt_path, device)
    kernel_params = model.collect_kernel_params()
    (out_dir / "kernel_params.json").write_text(json.dumps(kernel_params, indent=2))

    ds = STEADPickingDataset(args.split, seq_len=args.seq_len, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_examples, len(ds)), replace=False)

    records = []
    for i, idx in enumerate(indices, start=1):
        sample = ds[int(idx)]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}
        p_idx_guess = int(sample["p_idx"].item()) if sample["p_valid"] > 0 else int(
            torch.sigmoid(model(batch["x"], batch["t"])["p"][0]).argmax().item()
        )
        with torch.no_grad():
            outputs = model.forward_explain(
                batch["x"],
                batch["t"],
                include_kernel_row=True,
                kernel_row_idx=p_idx_guess,
                kernel_branch="p",
            )
        meta = plot_example(
            out_dir / f"explain_{i}.png",
            batch,
            outputs,
            kernel_params,
            args.pick_threshold,
            title=f"{args.split} idx={idx}",
        )
        meta["dataset_idx"] = int(idx)
        records.append(meta)

    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "kernel_params": kernel_params,
        "examples": records,
        "interpretation_notes": {
            "rho(t)": "Learned medium density; higher values increase attenuation in the kernel.",
            "wave_energy": "Shared-layer propagation energy used for event detection.",
            "p/s_envelope": "P/S branch wave-field envelope before the picking head.",
            "kernel_contrib": "Absolute row of the last P-branch Huygens kernel at pick time (causal past sources).",
            "learned_params": "gamma (envelope width), omega (phase frequency), wave_speed (light-cone slope).",
        },
    }
    (out_dir / "explain_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[explain] saved {len(records)} figures to {out_dir}")


if __name__ == "__main__":
    main()
