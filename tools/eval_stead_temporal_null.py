#!/usr/bin/env python
"""B2 — STEAD temporal nulls for NMI Paper B (time shuffle / reverse / block).

IMPORTANT: STEAD tensors are ``x: (B, T, C)`` with C=3 components — NOT (B, C, T).
The first B2 run shuffled the wrong axis (channels), which is why F1 barely moved.

  python tools/eval_stead_temporal_null.py --device cuda \\
      --checkpoint outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    picking_score,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)
from hnf.picking_model import build_picking_model, load_picking_model_state
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.train_stead_picking import move_batch_to_device, _model_forward


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stead/temporal_null_b2v2"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=0, help="0 = full test")
    p.add_argument("--block-size", type=int, default=40, help="block shuffle length (samples)")
    return p.parse_args()


def _assert_btc(x: torch.Tensor) -> None:
    # STEAD picking: (B, T, C) with C in {1,3} and T >> C
    if x.dim() != 3:
        raise ValueError(f"expected x (B,T,C), got {tuple(x.shape)}")
    if x.size(-1) > x.size(1):
        raise ValueError(
            f"x looks like (B,C,T)={tuple(x.shape)}; STEAD must be (B,T,C). "
            "Refusing to shuffle wrong axis."
        )


def _corrupt_batch(batch: dict, mode: str, *, block_size: int) -> dict:
    """Corrupt the **time** axis of waveform ``x: (B, T, C)``; labels unchanged."""
    out = dict(batch)
    x = batch["x"]
    _assert_btc(x)
    if mode == "clean":
        return out
    b, t_len, _c = x.shape
    if mode == "time_shuffle":
        x2 = torch.empty_like(x)
        for i in range(b):
            perm = torch.randperm(t_len, device=x.device)
            x2[i] = x[i, perm, :]
        out["x"] = x2
        return out
    if mode == "time_reverse":
        # Reverse waveform; keep ascending t → kernel still “forward” on reversed content
        out["x"] = torch.flip(x, dims=[1])
        return out
    if mode == "time_reverse_retimed":
        # Reverse both waveform and time axis (full time-reversal of the trajectory)
        out["x"] = torch.flip(x, dims=[1])
        t = batch["t"]
        out["t"] = torch.flip(t, dims=[1]).clone()
        # re-monotonicize labels? keep GT indices in original coordinates → should fail hard
        return out
    if mode == "block_shuffle":
        bs = max(1, min(int(block_size), t_len))
        n_blocks = (t_len + bs - 1) // bs
        x2 = torch.empty_like(x)
        for i in range(b):
            order = torch.randperm(n_blocks, device=x.device)
            chunks = []
            for bi in order.tolist():
                lo = bi * bs
                hi = min(t_len, lo + bs)
                chunks.append(x[i, lo:hi])
            x2[i] = torch.cat(chunks, dim=0)[:t_len]
        out["x"] = x2
        return out
    if mode == "circular_shift":
        # Large random roll — preserves local waveform stats, breaks absolute onset time
        x2 = torch.empty_like(x)
        for i in range(b):
            shift = int(torch.randint(t_len // 4, 3 * t_len // 4, (1,), device=x.device))
            x2[i] = torch.roll(x[i], shifts=shift, dims=0)
        out["x"] = x2
        return out
    raise ValueError(mode)


@torch.no_grad()
def evaluate_null(
    model, loader, device, *, mode: str, seq_len: int, cfg: dict, max_batches: int, block_size: int
):
    model.eval()
    acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, cfg.get("pick_tolerance_sec", 0.5))
    pick_threshold = float(cfg.get("pick_threshold", 0.3))
    n_seen = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        batch = _corrupt_batch(batch, mode, block_size=block_size)
        outputs = _model_forward(model, batch)

        det_pred = det_pred_from_logits(outputs["det"])
        det_true = batch["det"] > 0.5
        update_detection_counts(acc, det_pred, det_true)

        p_probs = torch.sigmoid(outputs["p"])
        s_probs = torch.sigmoid(outputs["s"])
        if cfg.get("post_process_p_before_s", False):
            p_probs, s_probs = apply_p_before_s_constraint(
                p_probs, s_probs, pick_threshold
            )

        for head_name, idx_name, valid_name, counts in [
            ("p", "p_idx", "p_valid", acc.p),
            ("s", "s_idx", "s_valid", acc.s),
        ]:
            probs = p_probs if head_name == "p" else s_probs
            # Always use batch GT indices (do not trust echoed outputs["p_idx"])
            gt_idx = batch[idx_name]
            update_picking_counts(
                counts,
                probs,
                det_pred,
                det_true,
                batch[valid_name] > 0,
                gt_idx,
                pick_threshold,
                tol,
                seq_len,
            )
        n_seen += batch["x"].size(0)

    metrics = finalize_metrics(acc)
    metrics["n"] = n_seen
    metrics["mode"] = mode
    metrics["pick_focus_score"] = picking_score(metrics, mode="pick_focus")
    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("args", {}))
    model = build_picking_model(
        embed_dim=cfg.get("embed_dim", 64),
        num_shared_layers=cfg.get("num_shared_layers", 2),
        num_branch_layers=cfg.get("num_branch_layers", 2),
        gamma=cfg.get("gamma", 0.5),
        omega=cfg.get("omega", 0.3),
        vp=cfg.get("vp", 8.0),
        vs=cfg.get("vs", 4.5),
        local_window_sec=cfg.get("local_window_sec", 15.0),
        dropout=cfg.get("dropout", 0.1),
        per_time_det=cfg.get("per_time_det", False),
        pick_head_hidden=cfg.get("pick_head_hidden", 24),
        pick_head_kernel=cfg.get("pick_head_kernel", 7),
        pick_head_layers=cfg.get("pick_head_layers", 3),
        multi_scale=cfg.get("multi_scale", False),
        sparse_band=cfg.get("sparse_band", False),
        num_anchors=int(cfg.get("num_anchors", 0)),
        residual_pick_head=cfg.get("residual_pick_head", True),
        residual_det_head=cfg.get("residual_det_head", True),
        enhanced_det_head=cfg.get("enhanced_det_head", False),
        noise_cancel=cfg.get("noise_cancel", False),
        noise_source_dim=cfg.get("noise_source_dim", 16),
        noise_det_pick_split=cfg.get("noise_det_pick_split", False),
        noise_pick_cues=cfg.get("noise_pick_cues", False),
        principle=cfg.get("principle", "huygens"),
        obliquity_scale=float(cfg.get("obliquity_scale", 1.0)),
    ).to(device)
    load_picking_model_state(model, ckpt["state_dict"], strict=False)

    seq_len = int(cfg.get("seq_len", 400))
    test_ds = STEADPickingDataset(
        "test",
        seq_len=seq_len,
        label_sigma_sec=cfg.get("label_sigma_sec", 0.4),
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    modes = (
        "clean",
        "time_shuffle",
        "block_shuffle",
        "circular_shift",
        "time_reverse",
        "time_reverse_retimed",
    )
    rows = []
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        m = evaluate_null(
            model, loader, device, mode=mode, seq_len=seq_len, cfg=cfg,
            max_batches=args.max_batches, block_size=args.block_size,
        )
        rows.append(m)
        print(
            f"  det_f1={m['det_f1']:.4f} p_f1={m['p_f1']:.4f} "
            f"s_f1={m['s_f1']:.4f} pick_focus={m['pick_focus_score']:.4f} n={m['n']}",
            flush=True,
        )

    by_id = {r["mode"]: r for r in rows}
    clean_s = by_id["clean"]["pick_focus_score"]
    shuf_s = by_id["time_shuffle"]["pick_focus_score"]
    drop = clean_s - shuf_s
    gate = (drop >= 0.10) or (clean_s > 1e-6 and drop / clean_s >= 0.25)

    # Secondary: at least one strong null should also drop ≥0.10
    secondary = {
        m: clean_s - by_id[m]["pick_focus_score"]
        for m in ("block_shuffle", "circular_shift", "time_reverse", "time_reverse_retimed")
    }
    secondary_pass = any(d >= 0.10 for d in secondary.values())

    summary = {
        "checkpoint": str(args.checkpoint),
        "x_layout": "(B, T, C) — time axis = dim 1",
        "note": "v2 fixes wrong-axis shuffle bug from temporal_null_b2",
        "clean_pick_focus": clean_s,
        "time_shuffle_pick_focus": shuf_s,
        "delta_shuffle": drop,
        "pass_time_shuffle_gate": gate,
        "secondary_deltas": secondary,
        "pass_any_strong_null": secondary_pass,
        "pass_criteria": "Δ(mean P/S F1) ≥ 0.10 OR relative drop ≥ 25% on time_shuffle",
        "rows": rows,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))

    def _row(mode: str) -> str:
        s = by_id[mode]["pick_focus_score"]
        return f"| {mode} | {s:.4f} | {s - clean_s:+.4f} |"

    lines = [
        "# STEAD temporal null v2 (NMI Paper B / B2)",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        "Layout: `x = (B, T, C)` — shuffle **time** (dim 1). Prior FAIL was a channel-axis bug.",
        "",
        "| Condition | mean P/S F1 | Δ vs clean |",
        "|---|---:|---:|",
        f"| clean | **{clean_s:.4f}** | — |",
    ]
    for mode in modes[1:]:
        lines.append(_row(mode))
    lines += [
        "",
        f"- time_shuffle gate: **{'PASS' if gate else 'FAIL'}** (Δ={drop:+.4f})",
        f"- any strong null ≥0.10 drop: **{'PASS' if secondary_pass else 'FAIL'}**",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
