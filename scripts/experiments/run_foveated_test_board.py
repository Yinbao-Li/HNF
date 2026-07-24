#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal STEAD test board: dense run28 (800) vs foveated zero-shot (6000, ≤8 gazes).

Also writes multi-sample gaze trajectory panels to docs/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.foveated.engine import visualize_trajectory_ascii
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    tolerance_bins,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_foveated import build_engine, collate_foveated
from tools.train_stead_picking import move_batch_to_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense vs foveated STEAD test board")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--output-dir", default="outputs/foveated/test_board")
    p.add_argument("--max-events", type=int, default=800, help="Cap test events (0=all)")
    p.add_argument("--max-gazes", type=int, default=8)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--n-traj", type=int, default=6, help="Trajectory panel samples")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def eval_dense(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    seq_len: int,
    pick_threshold: float,
    tol_bins: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    acc = EvalAccumulator()
    t0 = time.time()
    n = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        x, t = batch["x"], batch["t"]
        out = model(x, t)
        p_prob = torch.sigmoid(out["p"])
        s_prob = torch.sigmoid(out["s"])
        p_prob, s_prob = apply_p_before_s_constraint(p_prob, s_prob, pick_threshold)
        det_pred = torch.ones(x.size(0), dtype=torch.bool, device=device)
        update_picking_counts(
            acc.p, p_prob, det_pred, batch["det"] > 0.5, batch["p_valid"] > 0.5,
            batch["p_idx"], pick_threshold, tol_bins, seq_len,
        )
        update_picking_counts(
            acc.s, s_prob, det_pred, batch["det"] > 0.5, batch["s_valid"] > 0.5,
            batch["s_idx"], pick_threshold, tol_bins, seq_len,
        )
        n += x.size(0)
    m = finalize_metrics(acc)
    m["n"] = n
    m["sec"] = time.time() - t0
    m["sec_per_trace"] = m["sec"] / max(n, 1)
    return m


@torch.no_grad()
def eval_foveated_board(
    engine,
    loader: DataLoader,
    *,
    max_gazes: int,
    pick_threshold: float,
    tol_bins: int,
    device: torch.device,
) -> dict[str, float]:
    engine.eval()
    acc = EvalAccumulator()
    t0 = time.time()
    n = 0
    n_gazes = 0.0
    cover = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        wave = batch["wave_b3t"]
        out = engine(wave, max_gazes=max_gazes)
        p_prob, s_prob = apply_p_before_s_constraint(out.p_prob, out.s_prob, pick_threshold)
        det_pred = torch.ones(wave.size(0), dtype=torch.bool, device=device)
        update_picking_counts(
            acc.p, p_prob, det_pred, batch["det"] > 0.5, batch["p_valid"] > 0.5,
            batch["p_idx"], pick_threshold, tol_bins, engine.seq_len,
        )
        update_picking_counts(
            acc.s, s_prob, det_pred, batch["det"] > 0.5, batch["s_valid"] > 0.5,
            batch["s_idx"], pick_threshold, tol_bins, engine.seq_len,
        )
        n_gazes += float(out.n_gazes.float().sum().item())
        if out.coverage is not None:
            cover += float(out.coverage.mean().item()) * wave.size(0)
        n += wave.size(0)
    m = finalize_metrics(acc)
    m["n"] = n
    m["n_gazes_mean"] = n_gazes / max(n, 1)
    m["coverage_mean"] = cover / max(n, 1)
    m["sec"] = time.time() - t0
    m["sec_per_trace"] = m["sec"] / max(n, 1)
    return m


def save_trajectory_panel(
    engine,
    ds: STEADPickingDataset,
    out_path: Path,
    *,
    n_traj: int,
    max_gazes: int,
    device: torch.device,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n_traj, len(ds)), replace=False)
    fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 2.2 * len(idxs)), sharex=True)
    if len(idxs) == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        item = ds[int(idx)]
        x = item["x"].unsqueeze(0).to(device)
        wave = x.transpose(1, 2).contiguous()
        with torch.no_grad():
            out = engine(wave, max_gazes=max_gazes)
        t = item["t"][:, 0].numpy()
        z = item["x"][:, 2].numpy()
        ax.plot(t, z, lw=0.55, color="0.35")
        p_gt, s_gt = int(item["p_idx"]), int(item["s_idx"])
        if p_gt >= 0:
            ax.axvline(p_gt / engine.fovea.sample_rate_hz, color="green", ls="--", lw=1.1, label="GT P")
        if s_gt >= 0:
            ax.axvline(s_gt / engine.fovea.sample_rate_hz, color="blue", ls="--", lw=1.1, label="GT S")
        traj = out.trajectory[0]
        for i, node in enumerate(traj):
            ts = node["time_stamp"] / engine.fovea.sample_rate_hz
            ax.axvline(ts, color="crimson", alpha=0.45, lw=1.0)
            ymax = float(np.nanmax(np.abs(z))) if z.size else 1.0
            ax.text(ts, 0.85 * ymax, str(i), color="crimson", fontsize=7, ha="center")
        # predicted picks
        pp = int(out.p_idx[0]); sp = int(out.s_idx[0])
        ax.axvline(pp / engine.fovea.sample_rate_hz, color="darkorange", ls=":", lw=1.2, label="Pred P")
        ax.axvline(sp / engine.fovea.sample_rate_hz, color="purple", ls=":", lw=1.2, label="Pred S")
        ascii_line = visualize_trajectory_ascii(traj, seq_len=engine.seq_len, width=60)
        ax.set_ylabel(f"#{idx}")
        ax.set_title(
            f"gazes={int(out.n_gazes[0])}  {ascii_line}",
            fontsize=8,
            loc="left",
        )
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=4)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Foveated gaze trajectories (max_gazes={max_gazes})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    docs = _REPO_ROOT / "docs" / "figures" / "foveated_gaze_trajectory_panel.png"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_bytes(out_path.read_bytes())


def write_md(report: dict, path: Path) -> None:
    d = report["dense"]
    f = report["foveated"]
    lines = [
        "# STEAD Test: Dense run28 vs Foveated",
        "",
        f"- n_events: **{report['n']}** (split=test)",
        f"- backbone: `{report['checkpoint']}`",
        f"- pick threshold: {report['pick_threshold']}, tol: 0.5 s",
        "",
        "| Model | P-F1 | S-F1 | P-MAE | S-MAE | gazes | sec/trace |",
        "|-------|-----:|-----:|------:|------:|------:|----------:|",
        f"| Dense run28 (seq=800) | {d['p_f1']:.3f} | {d['s_f1']:.3f} | {d['p_mae_sec']:.3f} | {d['s_mae_sec']:.3f} | — | {d['sec_per_trace']:.3f} |",
        f"| Foveated ZS (≤{report['max_gazes']}) | {f['p_f1']:.3f} | {f['s_f1']:.3f} | {f['p_mae_sec']:.3f} | {f['s_mae_sec']:.3f} | {f['n_gazes_mean']:.2f} | {f['sec_per_trace']:.3f} |",
        "",
        f"- foveated coverage mean: {f.get('coverage_mean', 0):.3f}",
        f"- figure: `docs/figures/foveated_gaze_trajectory_panel.png`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    max_ev = None if args.max_events <= 0 else args.max_events

    # --- Dense @800 ---
    print("[board] dense run28 @800 …", flush=True)
    model, _ = load_model(Path(args.checkpoint), device)
    ds800 = STEADPickingDataset(
        "test", seq_len=800, max_event_traces=max_ev, max_noise_traces=0, seed=args.seed
    )
    # events only for fair pick board
    loader800 = DataLoader(ds800, batch_size=16, shuffle=False, num_workers=0)
    dense_m = eval_dense(
        model, loader800, seq_len=800, pick_threshold=args.pick_threshold,
        tol_bins=tolerance_bins(800, 0.5), device=device,
    )
    print(
        f"  dense: P={dense_m['p_f1']:.3f} S={dense_m['s_f1']:.3f} "
        f"({dense_m['sec_per_trace']:.3f}s/trace, n={dense_m['n']})",
        flush=True,
    )

    # --- Foveated @6000 ---
    print(f"[board] foveated ZS max_gazes={args.max_gazes} …", flush=True)

    class A:
        pass

    a = A()
    a.checkpoint = args.checkpoint
    a.seq_len = 6000
    a.max_gazes = args.max_gazes
    a.freeze_backbone = True
    a.unfreeze_backbone = False
    a.scanner = "energy"
    engine = build_engine(a, device)
    engine.coverage_complete_ratio = 0.85
    engine.eval()

    ds6 = STEADPickingDataset(
        "test", seq_len=6000, max_event_traces=max_ev, max_noise_traces=0, seed=args.seed
    )
    loader6 = DataLoader(ds6, batch_size=1, shuffle=False, collate_fn=collate_foveated)
    fov_m = eval_foveated_board(
        engine, loader6, max_gazes=args.max_gazes, pick_threshold=args.pick_threshold,
        tol_bins=tolerance_bins(6000, 0.5), device=device,
    )
    print(
        f"  foveated: P={fov_m['p_f1']:.3f} S={fov_m['s_f1']:.3f} "
        f"gazes={fov_m['n_gazes_mean']:.2f} ({fov_m['sec_per_trace']:.3f}s/trace)",
        flush=True,
    )

    print("[board] trajectory panel …", flush=True)
    fig_path = out / "foveated_gaze_trajectory_panel.png"
    save_trajectory_panel(
        engine, ds6, fig_path, n_traj=args.n_traj, max_gazes=args.max_gazes,
        device=device, seed=args.seed,
    )

    report = {
        "checkpoint": args.checkpoint,
        "split": "test",
        "n": int(dense_m["n"]),
        "pick_threshold": args.pick_threshold,
        "max_gazes": args.max_gazes,
        "dense": dense_m,
        "foveated": fov_m,
    }
    (out / "test_board.json").write_text(json.dumps(report, indent=2))
    write_md(report, out / "test_board.md")
    print(f"[board] wrote {out / 'test_board.md'}", flush=True)


if __name__ == "__main__":
    main()
