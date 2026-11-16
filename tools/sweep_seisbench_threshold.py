#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Val-selected pick threshold for SeisBench EQT / PhaseNet (same val set as c2f sweep).

  python tools/sweep_seisbench_threshold.py --models eqt,phasenet \
      --output-dir outputs/paper_stead_thresh_protocol
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO, _REPO / "scripts" / "paper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hnf.picking_metrics import (  # noqa: E402
    EvalAccumulator,
    apply_p_before_s_constraint,
    finalize_metrics,
    update_detection_counts,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from run_paper_obs_picking_compare import normalize_wave  # noqa: E402
from run_paper_stead_triple_compare import _load_raw_stead, eval_seisbench, load_sb  # noqa: E402

MODEL_MAP = {
    "eqt": ("EQT(STEAD)", "EQTransformer", "stead", "eqt"),
    "phasenet": ("PhaseNet(STEAD)", "PhaseNet", "stead", "phasenet"),
}
NATIVE_LEN = 6000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="eqt,phasenet")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paper_stead_thresh_protocol"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-val", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thresholds", default="0.10,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--skip-test", action="store_true")
    return p.parse_args()


@torch.no_grad()
def cache_val(model, ds, device, kind, batch_size):
    """Return fp16 CPU tensors (det_prob [N] or None, p [N,T], s [N,T]) + labels."""
    model = model.to(device).eval()
    labels = "".join(getattr(model, "labels", "PSN") or "PSN").upper()
    dets, ps, ss = [], [], []
    y = {"p_idx": [], "s_idx": [], "p_valid": [], "s_valid": [], "is_event": []}
    for start in range(0, len(ds), batch_size):
        samples = [_load_raw_stead(ds, i) for i in range(start, min(start + batch_size, len(ds)))]
        x = torch.from_numpy(np.stack([normalize_wave(s["wave"][::-1], "peak") for s in samples])).to(device)
        for k in y:
            y[k] += [s[k] for s in samples]
        out = model(x)
        if kind == "eqt":
            det, p, s = (t.squeeze(1) if t.dim() == 3 else t for t in out)
            dets.append(det.amax(-1).float().cpu())
        else:
            if labels.startswith("NP"):
                p, s = out[:, 1], out[:, 2]
            else:
                p, s = out[:, 0], out[:, 1]
        ps.append(p.half().cpu())
        ss.append(s.half().cpu())
    det = torch.cat(dets) if dets else None
    return det, torch.cat(ps), torch.cat(ss), {k: torch.tensor(v) for k, v in y.items()}


def metrics_at(det, p_all, s_all, y, kind, pick_th, det_th, tol, device, chunk=1024):
    acc = EvalAccumulator()
    for a in range(0, p_all.size(0), chunk):
        b = a + chunk
        p = p_all[a:b].to(device).float()
        s = s_all[a:b].to(device).float()
        yt = {k: v[a:b].to(device) for k, v in y.items()}
        if kind == "eqt":
            det_pred = det[a:b].to(device) >= det_th
        else:
            det_pred = torch.maximum(p.amax(-1), s.amax(-1)) >= pick_th
        p, s = apply_p_before_s_constraint(p, s, pick_th)
        update_detection_counts(acc, det_pred, yt["is_event"])
        update_picking_counts(acc.p, p, det_pred, yt["is_event"], yt["p_valid"], yt["p_idx"], pick_th, tol, NATIVE_LEN)
        update_picking_counts(acc.s, s, det_pred, yt["is_event"], yt["s_valid"], yt["s_idx"], pick_th, tol, NATIVE_LEN)
    m = finalize_metrics(acc)
    m["pick_focus"] = 0.5 * (m["p_f1"] + m["s_f1"])
    m["pick_threshold"] = pick_th
    return m


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ths = [float(t) for t in args.thresholds.split(",")]
    tol = max(1, int(round(args.tol_sec * 100.0)))

    val_ds = STEADPickingDataset(
        "val", seq_len=NATIVE_LEN,
        max_event_traces=args.max_val // 2, max_noise_traces=args.max_val // 2,
        seed=args.seed, load_geometry=False,
    )
    test_ds = None if args.skip_test else STEADPickingDataset("test", seq_len=NATIVE_LEN, load_geometry=False)
    print(f"[sb-sweep] val={len(val_ds)} test={0 if test_ds is None else len(test_ds)}", flush=True)

    report = {"tol_sec": args.tol_sec, "det_threshold": args.det_threshold, "models": {}}
    out_json = args.output_dir / "seisbench_threshold_sweep.json"
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        label, cls_name, weights, kind = MODEL_MAP[key]
        model = load_sb(cls_name, weights)
        t0 = time.time()
        det, p, s, y = cache_val(model, val_ds, device, kind, args.batch_size)
        rows = []
        for th in ths:
            m = metrics_at(det, p, s, y, kind, th, args.det_threshold, tol, device)
            rows.append(m)
            print(
                f"  {label} val th={th:.2f} det={m['det_f1']:.4f} P={m['p_f1']:.4f} "
                f"S={m['s_f1']:.4f} focus={m['pick_focus']:.4f}",
                flush=True,
            )
        best = max(rows, key=lambda r: r["pick_focus"])
        entry = {"val_sweep": rows, "selected_threshold": best["pick_threshold"], "best_val": best,
                 "val_sec": time.time() - t0}
        print(f"[sb-sweep] {label} selected th={best['pick_threshold']:.2f} val focus={best['pick_focus']:.4f}",
              flush=True)
        del det, p, s
        if test_ds is not None:
            t1 = time.time()
            te = eval_seisbench(
                model, test_ds, list(range(len(test_ds))), device, kind,
                pick_th=best["pick_threshold"], det_th=args.det_threshold,
                tol_sec=args.tol_sec, batch_size=args.batch_size,
            )["coupled"]
            te["pick_focus"] = 0.5 * (te["p_f1"] + te["s_f1"])
            entry["test_selected"] = te
            print(
                f"[sb-sweep] {label} test th={best['pick_threshold']:.2f} det={te['det_f1']:.4f} "
                f"P={te['p_f1']:.4f} S={te['s_f1']:.4f} focus={te['pick_focus']:.4f} "
                f"P_MAE={te['p_mae_sec']:.4f}s S_MAE={te['s_mae_sec']:.4f}s ({time.time() - t1:.0f}s)",
                flush=True,
            )
        report["models"][label] = entry
        out_json.write_text(json.dumps(report, indent=2))
    print(f"[done] → {out_json}", flush=True)


if __name__ == "__main__":
    main()
