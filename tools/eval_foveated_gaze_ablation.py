#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablate max_gazes (including a near-unlimited budget) on frozen run28 fovea.

Compares pick F1 / MAE / mean gaze count for budgets like 1,2,4,8,16,32
and an "unlimited" setting (large cap + coverage early-stop).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_metrics import tolerance_bins
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.train_foveated import build_engine, collate_foveated, eval_foveated
from tools.train_stead_picking import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Foveated gaze-budget ablation")
    p.add_argument("--backbone-checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--output-dir", default="outputs/foveated/gaze_ablation")
    p.add_argument("--budgets", default="1,2,4,8,16,32,128")
    p.add_argument("--unlimited-cap", type=int, default=128, help="Cap used as 'unlimited'")
    p.add_argument("--max-val", type=int, default=200)
    p.add_argument("--coverage-complete-ratio", type=float, default=0.85)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    class A:
        pass

    a = A()
    a.checkpoint = args.backbone_checkpoint
    a.seq_len = 6000
    a.max_gazes = args.unlimited_cap
    a.freeze_backbone = True
    a.unfreeze_backbone = False
    a.scanner = "energy"
    engine = build_engine(a, device)
    engine.coverage_complete_ratio = args.coverage_complete_ratio
    engine.eval()

    ds = STEADPickingDataset(
        "val",
        seq_len=6000,
        max_event_traces=args.max_val,
        max_noise_traces=0,
        augment=False,
        seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_foveated)
    tol = tolerance_bins(6000, 0.5)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    rows = []
    for budget in budgets:
        t0 = time.time()
        # Temporarily set engine default; eval_foveated calls engine(wave) without max_gazes
        # so we monkey-patch via a thin wrapper.
        engine.max_gazes = budget

        def _eval_with_budget():
            from tools.train_foveated import move_batch_to_device
            from hnf.picking_metrics import (
                EvalAccumulator,
                apply_p_before_s_constraint,
                finalize_metrics,
                update_picking_counts,
            )

            acc = EvalAccumulator()
            n_gazes_sum = 0.0
            n_samples = 0
            cover_sum = 0.0
            with torch.no_grad():
                for batch in loader:
                    batch = move_batch_to_device(batch, device)
                    wave = batch["wave_b3t"]
                    out = engine(wave, max_gazes=budget)
                    det_pred = torch.ones(wave.size(0), dtype=torch.bool, device=device)
                    p_prob, s_prob = apply_p_before_s_constraint(
                        out.p_prob, out.s_prob, args.pick_threshold
                    )
                    update_picking_counts(
                        acc.p,
                        p_prob,
                        det_pred,
                        batch["det"] > 0.5,
                        batch["p_valid"] > 0.5,
                        batch["p_idx"],
                        args.pick_threshold,
                        tol,
                        6000,
                    )
                    update_picking_counts(
                        acc.s,
                        s_prob,
                        det_pred,
                        batch["det"] > 0.5,
                        batch["s_valid"] > 0.5,
                        batch["s_idx"],
                        args.pick_threshold,
                        tol,
                        6000,
                    )
                    n_gazes_sum += float(out.n_gazes.float().sum().item())
                    if out.coverage is not None:
                        cover_sum += float(out.coverage.mean().item()) * wave.size(0)
                    n_samples += wave.size(0)
            m = finalize_metrics(acc)
            m["n_gazes_mean"] = n_gazes_sum / max(n_samples, 1)
            m["coverage_mean"] = cover_sum / max(n_samples, 1)
            m["sec"] = time.time() - t0
            m["budget"] = budget
            m["label"] = "unlimited" if budget == args.unlimited_cap else str(budget)
            return m

        m = _eval_with_budget()
        rows.append(m)
        print(
            f"budget={m['label']:>9}  P={m['p_f1']:.3f} S={m['s_f1']:.3f}  "
            f"gazes={m['n_gazes_mean']:.2f}  cover={m['coverage_mean']:.2f}  "
            f"time={m['sec']:.1f}s"
        )

    report = {
        "n": len(ds),
        "coverage_complete_ratio": args.coverage_complete_ratio,
        "rows": rows,
    }
    (out / "gaze_ablation.json").write_text(json.dumps(report, indent=2))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    xs = [r["n_gazes_mean"] for r in rows]
    ax[0].plot(xs, [r["p_f1"] for r in rows], "o-", label="P-F1")
    ax[0].plot(xs, [r["s_f1"] for r in rows], "s-", label="S-F1")
    ax[0].set_xlabel("Mean gazes used")
    ax[0].set_ylabel("F1")
    ax[0].set_title("F1 vs mean gazes")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].bar([r["label"] for r in rows], [r["n_gazes_mean"] for r in rows], color="steelblue")
    ax[1].set_xlabel("Budget cap")
    ax[1].set_ylabel("Mean gazes used")
    ax[1].set_title("Early-stop vs budget")
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig_path = out / "gaze_ablation.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    docs = _REPO_ROOT / "docs" / "figures" / "foveated_gaze_ablation.png"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_bytes(fig_path.read_bytes())
    print(f"Wrote {out / 'gaze_ablation.json'} and {fig_path}")


if __name__ == "__main__":
    main()
