#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump per-window [tp,fp,fn] for STEAD/ETHZ/OBS cluster-bootstrap CIs.

Aligned with manuscript decoders (Phase A correction 2026-11-17):
  - ETHZ/OBS: tools.eval_shift_invariance.run_c2f (demean/std + linspace t)
              + tools.sweep_seisbench_threshold.metrics_at / update_picking_counts
  - STEAD: dataset-normalized batches + same picking-count rules as metrics_at

  python tools/phase2_dump_headline_hits.py --device cuda --output-dir results/phase2/2_7_ci
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
for sub in ("", "scripts/paper"):
    if str(_REPO / sub) not in sys.path:
        sys.path.insert(0, str(_REPO / sub))

from hnf.picking_metrics import apply_p_before_s_constraint  # noqa: E402
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.eval_ethz_cross import (  # noqa: E402
    GROUP_COL,
    P_COLS,
    S_COLS,
    T,
    cut_window,
    first_arrival,
    load_dataset,
)
from tools.eval_shift_invariance import run_c2f  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402
from tools.train_stead_picking import move_batch_to_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase2/2_7_ci"))
    p.add_argument("--sets", default="stead_full,ethz_test,obs_test")
    p.add_argument("--max-traces", type=int, default=0)
    p.add_argument("--p-min", type=float, default=5.0)
    p.add_argument("--p-max", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bandpass", default="1,45")
    return p.parse_args()


def per_trace_picking_counts(
    probs: torch.Tensor,
    det_pred: torch.Tensor,
    is_event: torch.Tensor,
    valid: torch.Tensor,
    gt_idx: torch.Tensor,
    pick_th: float,
    tol: int,
) -> np.ndarray:
    """Return (N,3) int counts [tp,fp,fn] matching update_picking_counts semantics."""
    max_prob, pred_idx = probs.max(dim=-1)
    n = probs.size(0)
    out = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        is_ev = bool(is_event[i].item())
        has_gt = bool(valid[i].item())
        detected = bool(det_pred[i].item())
        pred_exists = float(max_prob[i].item()) >= pick_th
        pred_i = int(pred_idx[i].item())
        if not is_ev:
            if pred_exists:
                out[i, 1] = 1  # fp
            continue
        if not has_gt:
            continue
        gt_i = int(gt_idx[i].item())
        within = pred_exists and abs(pred_i - gt_i) <= tol
        if not detected:
            out[i, 2] = 1  # fn
        elif within:
            out[i, 0] = 1  # tp
        else:
            out[i, 2] = 1  # fn (wrong or missing peak) — NOT fp
    return out


def f1_from_counts(counts: np.ndarray) -> float:
    tp, fp, fn = counts.sum(0)
    return float(2 * tp / max(1.0, 2 * tp + fp + fn))


def pack_counts(counts_by_tol: dict) -> dict:
    """Serialize for bootstrap: models.ours.p['0.5'] = {counts: [[tp,fp,fn], ...]}"""
    return {tol: {"counts": c.tolist()} for tol, c in counts_by_tol.items()}


@torch.no_grad()
def dump_stead(args, device):
    model, cfg, _ = load_model(args.ours, device)
    seq_len = int(cfg.get("fine_len", 6000))
    ds = STEADPickingDataset(
        "test", seq_len=seq_len, max_event_traces=10**9, max_noise_traces=10**9, seed=42
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    names = []
    p_counts = {"0.1": [], "0.5": []}
    s_counts = {"0.1": [], "0.5": []}
    n_done = 0
    for batch in loader:
        names.extend([str(x) for x in batch.get("trace_name", [""] * len(batch["x"]))])
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det = torch.sigmoid(out["det"].float())
        gate = (det >= args.det_threshold).float().unsqueeze(-1)
        p = torch.sigmoid(out["p"].float()) * gate
        s = torch.sigmoid(out["s"].float()) * gate
        # metrics_at applies p_before_s inside; match that
        p, s = apply_p_before_s_constraint(p, s, args.pick_threshold)
        det_pred = det >= args.det_threshold
        is_event = batch["det"] > 0.5
        for tol_name, tol in (("0.1", 10), ("0.5", 50)):
            p_counts[tol_name].append(
                per_trace_picking_counts(
                    p, det_pred, is_event, batch["p_valid"], batch["p_idx"], args.pick_threshold, tol
                )
            )
            s_counts[tol_name].append(
                per_trace_picking_counts(
                    s, det_pred, is_event, batch["s_valid"], batch["s_idx"], args.pick_threshold, tol
                )
            )
        n_done += p.size(0)
        if args.max_traces and n_done >= args.max_traces:
            break

    for k in p_counts:
        p_counts[k] = np.concatenate(p_counts[k], axis=0)
        s_counts[k] = np.concatenate(s_counts[k], axis=0)
        if args.max_traces:
            p_counts[k] = p_counts[k][: args.max_traces]
            s_counts[k] = s_counts[k][: args.max_traces]
    if args.max_traces:
        names = names[: args.max_traces]
    if not names or names[0] == "":
        names = [f"trace_{i}" for i in range(len(p_counts["0.5"]))]
    clusters = [n.split("_")[0] if "_" in n else n for n in names]
    uniq = sorted(set(clusters))
    mp = {u: i for i, u in enumerate(uniq)}
    cluster_id = [mp[c] for c in clusters]
    d = {
        "source": str(args.ours),
        "decoder": "stead_dataset_norm+metrics_at_counts",
        "n": int(len(p_counts["0.5"])),
        "n_clusters": len(uniq),
        "cluster_kind": "station_prefix",
        "cluster_id": cluster_id,
        "models": {
            "ours": {
                "p": pack_counts(p_counts),
                "s": pack_counts(s_counts),
            }
        },
        "manuscript_refs": {
            "ours_p_0.5": 0.988,
            "ours_s_0.5": 0.975,
            "ours_p_0.1": 0.942,
            "ours_s_0.1": 0.869,
        },
        "point_f1": {
            "p@0.1": f1_from_counts(p_counts["0.1"]),
            "p@0.5": f1_from_counts(p_counts["0.5"]),
            "s@0.1": f1_from_counts(s_counts["0.1"]),
            "s@0.5": f1_from_counts(s_counts["0.5"]),
        },
    }
    return d


@torch.no_grad()
def dump_sb_set(args, device, dataset_name: str):
    model, _, _ = load_model(args.ours, device)
    data = load_dataset(dataset_name, "test")
    rng = np.random.default_rng(args.seed)
    n = len(data) if args.max_traces <= 0 else min(args.max_traces, len(data))
    order = rng.permutation(len(data))[:n]

    waves, meta_rows = [], []
    n_dead = 0
    for i in order:
        w, meta = data.get_sample(int(i))
        p = first_arrival(meta, P_COLS)
        if p is None or w.shape[0] != 3:
            continue
        w = np.nan_to_num(np.asarray(w, dtype=np.float32))
        if dataset_name == "obs" and not (w.std(axis=1) > 0).all():
            n_dead += 1
            continue
        s = first_arrival(meta, S_COLS)
        p = int(round(p))
        tp = int(round(rng.uniform(args.p_min, args.p_max) * 100))
        win, _, _ = cut_window(w, p, tp)
        s_rel = None if s is None else int(round(s)) - p + tp
        s_ok = s_rel is not None and 50 <= s_rel <= T - 50 and s_rel > tp
        st = str(meta.get("station_code") or meta.get("trace_name") or meta.get("trace_id") or i)
        waves.append(win)
        meta_rows.append(
            {
                "p_idx": tp,
                "s_idx": s_rel if s_ok else 0,
                "s_valid": bool(s_ok),
                "station": st,
            }
        )

    waves = np.stack(waves)
    if args.bandpass:
        lo, hi = (float(x) for x in args.bandpass.split(","))
        sos = butter(4, [lo, hi], btype="bandpass", fs=100.0, output="sos")
        waves = sosfiltfilt(sos, waves, axis=-1).astype(np.float32)

    # Manuscript decoder
    det, p, s = run_c2f(model, waves, device, args.batch_size, args.det_threshold)
    # metrics_at applies p_before_s internally — replicate before count
    p = p.float()
    s = s.float()
    p, s = apply_p_before_s_constraint(p, s, args.pick_threshold)
    det_pred = det >= args.det_threshold
    is_event = torch.ones(len(waves), dtype=torch.bool)
    p_idx = torch.tensor([m["p_idx"] for m in meta_rows])
    s_idx = torch.tensor([m["s_idx"] for m in meta_rows])
    p_valid = torch.ones(len(waves), dtype=torch.bool)
    s_valid = torch.tensor([m["s_valid"] for m in meta_rows], dtype=torch.bool)

    p_counts, s_counts = {}, {}
    for tol_name, tol in (("0.1", 10), ("0.5", 50)):
        p_counts[tol_name] = per_trace_picking_counts(
            p, det_pred, is_event, p_valid, p_idx, args.pick_threshold, tol
        )
        s_counts[tol_name] = per_trace_picking_counts(
            s, det_pred, is_event, s_valid, s_idx, args.pick_threshold, tol
        )

    clusters = [m["station"] for m in meta_rows]
    uniq = sorted(set(clusters))
    mp = {u: i for i, u in enumerate(uniq)}
    cluster_id = [mp[c] for c in clusters]
    return {
        "source": str(args.ours),
        "dataset": dataset_name,
        "decoder": "run_c2f+metrics_at_counts",
        "n": int(len(waves)),
        "n_clusters": len(uniq),
        "n_dead_dropped": n_dead,
        "cluster_kind": "station",
        "cluster_id": cluster_id,
        "protocol": {
            "seed": args.seed,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "bandpass": args.bandpass,
            "normalize": "demean_std_per_channel",
            "time_axis": "linspace_0_60_BT1",
        },
        "models": {"ours": {"p": pack_counts(p_counts), "s": pack_counts(s_counts)}},
        "point_f1": {
            "p@0.1": f1_from_counts(p_counts["0.1"]),
            "p@0.5": f1_from_counts(p_counts["0.5"]),
            "s@0.1": f1_from_counts(s_counts["0.1"]),
            "s@0.5": f1_from_counts(s_counts["0.5"]),
        },
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for name in args.sets.split(","):
        print(f"[dump] {name} on {device}", flush=True)
        if name == "stead_full":
            d = dump_stead(args, device)
        elif name == "ethz_test":
            d = dump_sb_set(args, device, "ethz")
        elif name == "obs_test":
            d = dump_sb_set(args, device, "obs")
        else:
            raise ValueError(name)
        print(f"  point_f1={d.get('point_f1')} n={d['n']} clusters={d['n_clusters']}", flush=True)
        out = args.output_dir / f"{name}_hits.json"
        out.write_text(json.dumps(d))
        print(f"  -> {out}", flush=True)
    print("[done] dumps", flush=True)


if __name__ == "__main__":
    main()
