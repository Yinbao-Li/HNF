#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short multi-grid probe for a picking checkpoint.

Reports det/P/S F1 (+ MAE) on a fixed val slice after resampling each batch
onto each grid length. Use after run29 to check extrapolation beyond the
training grids (e.g. 200, 2000, 6000).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hnf.grid_augment import (
    clamp_kernel_windows_for_grid,
    parse_grid_lens,
    resample_batch_to_grid,
)
from hnf.picking_metrics import apply_p_before_s_constraint
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-grid invariance probe")
    p.add_argument(
        "--checkpoint",
        default="outputs/run29/29_grid_invariance_ft_v2/best.pt",
    )
    p.add_argument("--base-seq-len", type=int, default=800)
    p.add_argument(
        "--grid-lens",
        default="200,400,600,800,1000,2000,6000",
        help="comma-separated sample counts for the fixed 60 s window",
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-event", type=int, default=400)
    p.add_argument("--max-noise", type=int, default=200)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument(
        "--grid-max-band-bins",
        type=int,
        default=140,
        help="match training clamp; 0 disables",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output-dir",
        default="outputs/grid_probe_run29",
    )
    return p.parse_args()


def _peak(probs: torch.Tensor, thr: float) -> int | None:
    p = probs.reshape(-1)
    if float(p.max().item()) < thr:
        return None
    return int(p.argmax().item())


def _hit(pred: int | None, gt: int, tol: int) -> bool:
    if pred is None or gt < 0:
        return False
    return abs(pred - gt) <= tol


@torch.no_grad()
def eval_grid(
    model,
    loader,
    *,
    grid_len: int,
    pick_th: float,
    tol_sec: float,
    label_sigma_sec: float,
    device: torch.device,
    grid_max_band_bins: int = 0,
) -> dict:
    tp = {"p": 0, "s": 0}
    fp = {"p": 0, "s": 0}
    fn = {"p": 0, "s": 0}
    abs_err = {"p": [], "s": []}
    det_tp = det_fp = det_fn = 0
    n = 0
    if grid_max_band_bins > 0:
        clamp_kernel_windows_for_grid(
            model, grid_len, max_band_bins=grid_max_band_bins
        )
    t0 = time.perf_counter()
    for batch in loader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        batch = resample_batch_to_grid(
            batch, grid_len, label_sigma_sec=label_sigma_sec
        )
        x = batch["x"]
        t = batch["t"]
        if t.dim() == 3:
            t = t[0]
        out = model(x, t)
        p = torch.sigmoid(out["p"])
        s = torch.sigmoid(out["s"])
        p, s = apply_p_before_s_constraint(p, s, pick_th)
        det = torch.sigmoid(out["det"])
        if det.dim() > 1:
            det = det.amax(dim=-1)
        is_event = bool(batch["det"][0].item() > 0.5)
        det_pred = float(det[0].item()) >= 0.5
        if is_event and det_pred:
            det_tp += 1
        elif is_event and not det_pred:
            det_fn += 1
        elif (not is_event) and det_pred:
            det_fp += 1

        tol = max(1, int(round(tol_sec / 60.0 * (grid_len - 1))))
        for name, probs, key_idx in (
            ("p", p, "p_idx"),
            ("s", s, "s_idx"),
        ):
            pred = _peak(probs[0], pick_th)
            gt = int(batch[key_idx][0].item()) if key_idx in batch else -1
            if not is_event:
                if pred is not None:
                    fp[name] += 1
                continue
            if gt < 0:
                continue
            if _hit(pred, gt, tol):
                tp[name] += 1
                abs_err[name].append(abs(pred - gt) * 60.0 / (grid_len - 1))
            elif pred is None:
                fn[name] += 1
            else:
                fp[name] += 1
                fn[name] += 1
        n += 1
    elapsed = time.perf_counter() - t0

    def _f1(a, b, c):
        prec = a / max(a + b, 1)
        rec = a / max(a + c, 1)
        return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    return {
        "grid_len": grid_len,
        "hz": round((grid_len - 1) / 60.0, 3),
        "n": n,
        "det_f1": _f1(det_tp, det_fp, det_fn),
        "p_f1": _f1(tp["p"], fp["p"], fn["p"]),
        "s_f1": _f1(tp["s"], fp["s"], fn["s"]),
        "p_mae_sec": float(sum(abs_err["p"]) / max(len(abs_err["p"]), 1)),
        "s_mae_sec": float(sum(abs_err["s"]) / max(len(abs_err["s"]), 1)),
        "ms_per_trace": 1000.0 * elapsed / max(n, 1),
        "det_tp": det_tp,
        "det_fp": det_fp,
        "det_fn": det_fn,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, _ = load_model(Path(args.checkpoint), device)
    model.eval()

    ds = STEADPickingDataset(
        args.split,
        seq_len=args.base_seq_len,
        max_event_traces=args.max_event,
        max_noise_traces=args.max_noise,
        seed=2,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    grids = parse_grid_lens(args.grid_lens)
    rows = []
    print(
        f"[grid-probe] ckpt={args.checkpoint} n={len(ds)} grids={grids} "
        f"max_band_bins={args.grid_max_band_bins}"
    )
    for gl in grids:
        try:
            row = eval_grid(
                model,
                loader,
                grid_len=gl,
                pick_th=args.pick_threshold,
                tol_sec=args.tol_sec,
                label_sigma_sec=args.label_sigma_sec,
                device=device,
                grid_max_band_bins=args.grid_max_band_bins,
            )
        except RuntimeError as e:
            row = {"grid_len": gl, "error": str(e)[:300]}
            print(f"  L={gl}: ERROR {e}")
        else:
            print(
                f"  L={gl:5d} ({row['hz']:6.1f} Hz)  "
                f"det={row['det_f1']:.3f}  p={row['p_f1']:.3f}  s={row['s_f1']:.3f}  "
                f"p_mae={row['p_mae_sec']:.3f}s  s_mae={row['s_mae_sec']:.3f}s  "
                f"{row['ms_per_trace']:.1f} ms/tr"
            )
        rows.append(row)
        (out_dir / "report.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"[grid-probe] wrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
