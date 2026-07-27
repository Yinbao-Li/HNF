#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bar charts for 3D fluid model comparison (baselines + literature SOTA + HNF)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    board_path = _REPO / "outputs/fluid/baseline3d4d_board.json"
    if not board_path.is_file():
        raise SystemExit(f"missing {board_path}")
    board = json.loads(board_path.read_text())
    fig_dir = _REPO / "docs/figures/fluid"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- vortex_tube ---
    vortex_keys = [
        ("unet3d", "U-Net 3D", "#4C72B0"),
        ("recfno3d", "RecFNO3D", "#DD8452"),
        ("flowmri_net3d", "FlowMRI-Net3D", "#55A868"),
        ("hnf_spatial3d_rot", "HNF spatial3D", "#C44E52"),
    ]
    labels, vals, colors = [], [], []
    for key, label, color in vortex_keys:
        if key in board and "vel_rel" in board[key]:
            labels.append(label)
            vals.append(board[key]["vel_rel"])
            colors.append(color)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y, labels)
    ax.set_xlabel("test vel_rel (lower is better)")
    ax.set_title("3D vortex_tube @10% keep (30 ep)")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    # --- all families ---
    all_keys = [
        ("unet3d_all", "U-Net 3D", "#4C72B0"),
        ("hnf_spatial3d_all", "HNF spatial3D", "#C44E52"),
        ("recfno3d_all", "RecFNO3D", "#DD8452"),
        ("flowmri_net3d_all", "FlowMRI-Net3D", "#55A868"),
    ]
    labels2, vals2, colors2 = [], [], []
    for key, label, color in all_keys:
        if key in board and "vel_rel" in board[key]:
            labels2.append(label)
            vals2.append(board[key]["vel_rel"])
            colors2.append(color)

    ax2 = axes[1]
    y2 = np.arange(len(labels2))
    ax2.barh(y2, vals2, color=colors2, height=0.6)
    ax2.set_yticks(y2, labels2)
    ax2.set_xlabel("test vel_rel (lower is better)")
    ax2.set_title("3D all families @10% keep")
    ax2.invert_yaxis()
    for i, v in enumerate(vals2):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Sparse 3D velocity reconstruction — baselines vs literature SOTA vs HNF", fontsize=11)
    fig.tight_layout()
    out = fig_dir / "fluid3d_model_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] → {out}", flush=True)


if __name__ == "__main__":
    main()
