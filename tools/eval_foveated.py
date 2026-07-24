#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained FoveatedEngine on STEAD val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.foveated import FoveatedEngine, FoveaProcessor, PeripheralScanner, Scheduler
from hnf.foveated.engine import visualize_trajectory_ascii
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_foveated import build_engine, collate_foveated, eval_foveated
from tools.train_stead_picking import move_batch_to_device, set_seed
from hnf.picking_metrics import tolerance_bins


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval foveated engine")
    p.add_argument("--checkpoint", required=True, help="FoveatedEngine checkpoint")
    p.add_argument("--backbone-checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--max-val", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--pick-tolerance-sec", type=float, default=0.5)
    p.add_argument("--output", default="")
    p.add_argument("--save-trajectory-fig", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_foveated_engine(ckpt_path: Path, backbone_path: Path, device: torch.device) -> FoveatedEngine:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    class _Args:
        pass

    a = _Args()
    a.checkpoint = str(backbone_path)
    a.seq_len = int(saved_args.get("seq_len", 6000))
    a.max_gazes = int(saved_args.get("max_gazes", 8))
    a.freeze_backbone = True
    a.unfreeze_backbone = False
    a.scanner = saved_args.get("scanner", "energy")
    engine = build_engine(a, device)  # type: ignore[arg-type]
    engine.load_state_dict(ckpt["state_dict"], strict=False)
    engine.eval()
    return engine


def save_trajectory_figure(engine: FoveatedEngine, loader: DataLoader, out_path: Path, device: torch.device) -> None:
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        out = engine(batch["wave_b3t"])
    traj = out.trajectory[0]
    ascii_line = visualize_trajectory_ascii(traj, seq_len=engine.seq_len)

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    wave = batch["x"][0].cpu().numpy()
    t = batch["t"][0, :, 0].cpu().numpy()
    for ch, ax in enumerate(axes):
        ax.plot(t, wave[:, ch], lw=0.6, color="0.35")
        ax.set_ylabel(f"Ch{ch}")
    for node in traj:
        ts = node["time_stamp"] / engine.fovea.sample_rate_hz
        axes[0].axvline(ts, color="crimson", alpha=0.5, lw=1)
    p_gt = int(batch["p_idx"][0].item())
    s_gt = int(batch["s_idx"][0].item())
    if p_gt >= 0:
        axes[0].axvline(p_gt / engine.fovea.sample_rate_hz, color="green", ls="--", lw=1.2, label="GT P")
    if s_gt >= 0:
        axes[0].axvline(s_gt / engine.fovea.sample_rate_hz, color="blue", ls="--", lw=1.2, label="GT S")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"Gaze trajectory (n={len(traj)}): {ascii_line}", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    docs = _REPO_ROOT / "docs" / "figures" / "foveated_gaze_trajectory_sample.png"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_bytes(out_path.read_bytes())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    engine = load_foveated_engine(
        Path(args.checkpoint),
        Path(args.backbone_checkpoint),
        device,
    )
    ds = STEADPickingDataset(
        args.split,
        seq_len=engine.seq_len,
        max_event_traces=args.max_val,
        max_noise_traces=max(50, args.max_val // 4),
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_foveated)
    tol_bins = tolerance_bins(engine.seq_len, args.pick_tolerance_sec)
    metrics = eval_foveated(
        engine,
        loader,
        seq_len=engine.seq_len,
        pick_threshold=args.pick_threshold,
        tol_bins=tol_bins,
        device=device,
    )
    report = {"split": args.split, "n": len(ds), "metrics": metrics, "checkpoint": args.checkpoint}
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    if args.save_trajectory_fig:
        save_trajectory_figure(engine, loader, Path(args.save_trajectory_fig), device)


if __name__ == "__main__":
    main()
