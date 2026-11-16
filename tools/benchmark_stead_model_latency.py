#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-trace inference latency: c2f v3 vs run28 vs EQTransformer vs PhaseNet.

Forward-only on synthetic 60 s 3C windows (latency does not depend on content).

  python tools/benchmark_stead_model_latency.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-sizes", default="1,32")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--c2f-ckpt", type=Path, default=Path("outputs/stead_c2f_750_v3/best.pt"))
    p.add_argument("--run28-ckpt", type=Path,
                   default=Path("outputs/run28/28_ms_fresnel_phys_50ep_local/best.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/stead_c2f_750_v3/latency.json"))
    return p.parse_args()


def load_c2f(path: Path, device):
    from tools.sweep_stead_c2f_threshold import load_model
    model, _, _ = load_model(path, device)
    return model


def load_run28(path: Path, device):
    from hnf.picking_model import build_picking_model, load_picking_model_state
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = build_picking_model(
        embed_dim=a.get("embed_dim", 64),
        num_shared_layers=a.get("num_shared_layers", 2),
        num_branch_layers=a.get("num_branch_layers", 2),
        gamma=a.get("gamma", 0.5),
        omega=a.get("omega", 0.3),
        vp=a.get("vp", 8.0),
        vs=a.get("vs", 4.5),
        local_window_sec=a.get("local_window_sec", 15.0),
        dropout=a.get("dropout", 0.1),
        per_time_det=a.get("per_time_det", False),
        pick_head_hidden=a.get("pick_head_hidden", 24),
        pick_head_kernel=a.get("pick_head_kernel", 7),
        pick_head_layers=a.get("pick_head_layers", 3),
        multi_scale=a.get("multi_scale", False),
        sparse_band=a.get("sparse_band", False),
        num_anchors=int(a.get("num_anchors", 0)),
        residual_pick_head=a.get("residual_pick_head", True),
        residual_det_head=a.get("residual_det_head", True),
        enhanced_det_head=a.get("enhanced_det_head", False),
        noise_cancel=a.get("noise_cancel", False),
        noise_source_dim=a.get("noise_source_dim", 16),
        noise_det_pick_split=a.get("noise_det_pick_split", False),
        noise_pick_cues=a.get("noise_pick_cues", False),
        principle=a.get("principle", "huygens"),
        obliquity_scale=float(a.get("obliquity_scale", 1.0)),
    ).to(device)
    load_picking_model_state(model, ckpt["state_dict"], strict=False)
    return model, int(a.get("seq_len", 800))


@torch.no_grad()
def time_fn(fn, device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    args = parse_args()
    device = torch.device(args.device)
    bss = [int(x) for x in args.batch_sizes.split(",")]

    import seisbench.models as sbm

    models = {}
    c2f = load_c2f(args.c2f_ckpt, device).eval()
    models["HNF c2f v3 (6000)"] = (c2f, "hnf", 6000)
    r28, r28_len = load_run28(args.run28_ckpt, device)
    models[f"HNF run28 ({r28_len})"] = (r28.eval(), "hnf", r28_len)
    eqt = sbm.EQTransformer.from_pretrained("stead").to(device).eval()
    models["EQTransformer (6000)"] = (eqt, "sb", 6000)
    pn = sbm.PhaseNet.from_pretrained("stead").to(device).eval()
    models["PhaseNet (3001)"] = (pn, "sb", 3001)

    results = {}
    for name, (m, kind, T) in models.items():
        n_params = sum(p.numel() for p in m.parameters())
        results[name] = {"n_params": n_params, "seq_len": T, "ms": {}}
        for bs in bss:
            if kind == "hnf":
                x = torch.randn(bs, T, 3, device=device)
                t = torch.linspace(0.0, 60.0, T, device=device).view(1, T, 1).expand(bs, -1, -1)
                fn = lambda m=m, x=x, t=t: m(x, t)
            else:
                x = torch.randn(bs, 3, T, device=device)
                fn = lambda m=m, x=x: m(x)
            try:
                ms_batch = time_fn(fn, device, args.warmup, args.iters)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[name]["ms"][str(bs)] = "OOM"
                print(f"{name:26s} bs={bs:3d} OOM", flush=True)
                continue
            results[name]["ms"][str(bs)] = {
                "ms_per_batch": ms_batch,
                "ms_per_trace": ms_batch / bs,
            }
            print(
                f"{name:26s} params={n_params/1e3:7.1f}k bs={bs:3d} "
                f"{ms_batch:8.2f} ms/batch  {ms_batch/bs:7.3f} ms/trace",
                flush=True,
            )

    meta = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "note": "forward only, fp32, no post-processing; PhaseNet uses its native 30 s (3001) window",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"[done] → {args.output}")


if __name__ == "__main__":
    main()
