#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump per-window hit/pred/valid arrays for STEAD/ETHZ/OBS cluster-bootstrap CIs.

Matches manuscript eval protocols:
  - STEAD: STEADPickingDataset test + c2f hard det gate
  - ETHZ/OBS: same windowing as tools/eval_ethz_cross.py (seed, p_min/p_max, bandpass)

  python tools/phase2_dump_headline_hits.py --device cuda --output-dir results/phase2/2_7_ci
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

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
    p.add_argument("--max-traces", type=int, default=0, help="0=all")
    p.add_argument("--p-min", type=float, default=5.0)
    p.add_argument("--p-max", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bandpass", default="1,45")
    return p.parse_args()


def pack_phase(pred_idx, pred_ok, gt_idx, valid, tols_bins):
    out = {}
    for tol_name, tol in tols_bins.items():
        hit = pred_ok & valid & (np.abs(pred_idx - gt_idx) <= tol)
        out[tol_name] = {
            "hit": hit.astype(bool).tolist(),
            "pred": pred_ok.astype(bool).tolist(),
            "valid": valid.astype(bool).tolist(),
        }
    return out


def f1_print(d, label):
    for phase in ("p", "s"):
        for tol in ("0.1", "0.5"):
            arr = d["models"]["ours"][phase][tol]
            hit = np.array(arr["hit"])
            valid = np.array(arr["valid"])
            pred = np.array(arr["pred"])
            tp = (hit & valid & pred).sum()
            fp = (pred & ~hit).sum()
            fn = (valid & ~hit).sum()
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            print(f"  {label} {phase}@{tol}: F1={f1:.4f} n={d['n']} clusters={d['n_clusters']}", flush=True)


@torch.no_grad()
def dump_stead(args, device):
    model, cfg, _ = load_model(args.ours, device)
    seq_len = int(cfg.get("fine_len", 6000))
    ds = STEADPickingDataset(
        "test", seq_len=seq_len, max_event_traces=10**9, max_noise_traces=10**9, seed=42
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    p_max, p_idx, s_max, s_idx, det = [], [], [], [], []
    p_gt, s_gt, p_valid, s_valid, names = [], [], [], [], []
    for batch in loader:
        names.extend([str(x) for x in batch.get("trace_name", [""] * len(batch["x"]))])
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["t"])
        det_prob = torch.sigmoid(out["det"])
        gate = (det_prob >= args.det_threshold).float().unsqueeze(-1)
        p = torch.sigmoid(out["p"]) * gate
        s = torch.sigmoid(out["s"]) * gate
        p, s = apply_p_before_s_constraint(p, s, args.pick_threshold)
        pm, pi = p.max(-1)
        sm, si = s.max(-1)
        p_max.append(pm.cpu().numpy())
        p_idx.append(pi.cpu().numpy())
        s_max.append(sm.cpu().numpy())
        s_idx.append(si.cpu().numpy())
        det.append(det_prob.cpu().numpy())
        p_gt.append(batch["p_idx"].cpu().numpy())
        s_gt.append(batch["s_idx"].cpu().numpy())
        p_valid.append(batch["p_valid"].cpu().numpy())
        s_valid.append(batch["s_valid"].cpu().numpy())
        if args.max_traces and sum(len(x) for x in p_max) >= args.max_traces:
            break
    p_max = np.concatenate(p_max)
    p_idx = np.concatenate(p_idx)
    s_max = np.concatenate(s_max)
    s_idx = np.concatenate(s_idx)
    det = np.concatenate(det)
    p_gt = np.concatenate(p_gt)
    s_gt = np.concatenate(s_gt)
    p_valid = np.concatenate(p_valid).astype(bool)
    s_valid = np.concatenate(s_valid).astype(bool)
    if args.max_traces:
        p_max = p_max[: args.max_traces]
        p_idx = p_idx[: args.max_traces]
        s_max = s_max[: args.max_traces]
        s_idx = s_idx[: args.max_traces]
        det = det[: args.max_traces]
        p_gt = p_gt[: args.max_traces]
        s_gt = s_gt[: args.max_traces]
        p_valid = p_valid[: args.max_traces]
        s_valid = s_valid[: args.max_traces]
        names = names[: args.max_traces]
    if not names or names[0] == "":
        names = [f"trace_{i}" for i in range(len(p_max))]
    clusters = [n.split("_")[0] if "_" in n else n for n in names]
    uniq = sorted(set(clusters))
    mp = {u: i for i, u in enumerate(uniq)}
    cluster_id = [mp[c] for c in clusters]
    tols = {"0.1": 10, "0.5": 50}
    p_pred = (det >= args.det_threshold) & (p_max >= args.pick_threshold)
    s_pred = (det >= args.det_threshold) & (s_max >= args.pick_threshold)
    return {
        "source": str(args.ours),
        "n": int(len(p_max)),
        "n_clusters": len(uniq),
        "cluster_kind": "station_prefix",
        "cluster_id": cluster_id,
        "models": {
            "ours": {
                "p": pack_phase(p_idx, p_pred, p_gt, p_valid, tols),
                "s": pack_phase(s_idx, s_pred, s_gt, s_valid, tols),
            }
        },
        "manuscript_refs": {
            "ours_p_0.5": 0.988,
            "ours_s_0.5": 0.975,
            "ours_p_0.1": 0.942,
            "ours_s_0.1": 0.869,
        },
    }


@torch.no_grad()
def dump_sb_set(args, device, dataset_name: str):
    """Match eval_ethz_cross.py window protocol (seeded P placement + bandpass)."""
    model, cfg, _ = load_model(args.ours, device)
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
        g = meta.get(GROUP_COL[dataset_name])
        st = str(meta.get("station_code") or meta.get("trace_name") or meta.get("trace_id") or i)
        waves.append(win)
        meta_rows.append(
            {
                "p_idx": tp,
                "s_idx": s_rel if s_ok else 0,
                "s_valid": bool(s_ok),
                "group": str(g) if g else "unk",
                "station": st,
            }
        )

    waves = np.stack(waves)
    if args.bandpass:
        from scipy.signal import butter, sosfiltfilt

        lo, hi = (float(x) for x in args.bandpass.split(","))
        sos = butter(4, [lo, hi], btype="bandpass", fs=100.0, output="sos")
        waves = sosfiltfilt(sos, waves, axis=-1).astype(np.float32)

    # batched infer: model expects (B, T, 3) + time axis
    p_max, p_idx, s_max, s_idx, det = [], [], [], [], []
    for start in range(0, len(waves), args.batch_size):
        chunk = waves[start : start + args.batch_size]  # (B,3,T)
        x = torch.from_numpy(chunk).to(device).transpose(1, 2)  # (B,T,3)
        t = torch.arange(x.shape[1], device=device).float().unsqueeze(0).expand(x.shape[0], -1) * (
            60.0 / x.shape[1]
        )
        out = model(x, t)
        det_prob = torch.sigmoid(out["det"])
        gate = (det_prob >= args.det_threshold).float().unsqueeze(-1)
        p = torch.sigmoid(out["p"]) * gate
        s = torch.sigmoid(out["s"]) * gate
        p, s = apply_p_before_s_constraint(p, s, args.pick_threshold)
        pm, pi = p.max(-1)
        sm, si = s.max(-1)
        p_max.append(pm.cpu().numpy())
        p_idx.append(pi.cpu().numpy())
        s_max.append(sm.cpu().numpy())
        s_idx.append(si.cpu().numpy())
        det.append(det_prob.cpu().numpy())

    p_max = np.concatenate(p_max)
    p_idx = np.concatenate(p_idx)
    s_max = np.concatenate(s_max)
    s_idx = np.concatenate(s_idx)
    det = np.concatenate(det)
    p_gt = np.array([m["p_idx"] for m in meta_rows])
    s_gt = np.array([m["s_idx"] for m in meta_rows])
    s_valid = np.array([m["s_valid"] for m in meta_rows], dtype=bool)
    p_valid = np.ones(len(meta_rows), dtype=bool)
    clusters = [m["station"] for m in meta_rows]
    uniq = sorted(set(clusters))
    mp = {u: i for i, u in enumerate(uniq)}
    cluster_id = [mp[c] for c in clusters]
    tols = {"0.1": 10, "0.5": 50}
    p_pred = (det >= args.det_threshold) & (p_max >= args.pick_threshold)
    s_pred = (det >= args.det_threshold) & (s_max >= args.pick_threshold)

    # manuscript refs from final compare (ours rows)
    refs = {}
    if dataset_name == "ethz":
        refs = {"ours_p_0.5": None, "ours_s_0.5": None}  # filled from final JSON if needed
    return {
        "source": str(args.ours),
        "dataset": dataset_name,
        "n": int(len(p_max)),
        "n_clusters": len(uniq),
        "n_dead_dropped": n_dead,
        "cluster_kind": "station",
        "cluster_id": cluster_id,
        "protocol": {
            "seed": args.seed,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "bandpass": args.bandpass,
        },
        "models": {
            "ours": {
                "p": pack_phase(p_idx, p_pred, p_gt, p_valid, tols),
                "s": pack_phase(s_idx, s_pred, s_gt, s_valid, tols),
            }
        },
        "manuscript_refs": refs,
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
        f1_print(d, name)
        out = args.output_dir / f"{name}_hits.json"
        out.write_text(json.dumps(d))
        print(f"  -> {out}", flush=True)
    print("[done] dumps", flush=True)


if __name__ == "__main__":
    main()
