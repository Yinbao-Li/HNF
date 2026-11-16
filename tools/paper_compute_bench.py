#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parameters, inference throughput and peak memory of ours vs EQTransformer / PhaseNet on one GPU.

Inference only, batch 64, 6000 × 3 input, fp32 and bf16 autocast, no_grad; median of 3 timed runs.

  python tools/paper_compute_bench.py --output outputs/paper_tgrs/compute.json
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
sys.path.insert(0, str(_REPO))

from tools.eval_shift_invariance import load_sbc  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402


def bench(fn, x, bf16, iters):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        rates = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            torch.cuda.synchronize()
            rates.append(x.size(0) * iters / (time.perf_counter() - t0))
    return float(np.median(rates)), torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--eqt", type=Path, default=Path("outputs/sb_aug_eqt/last.pt"))
    p.add_argument("--pn", type=Path, default=Path("outputs/sb_aug_phasenet/last.pt"))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("outputs/paper_tgrs/compute.json"))
    a = p.parse_args()
    dev = torch.device("cuda")
    models = {"ours": (load_model(a.ours, dev)[0], True)}
    for name, path in (("EQTransformer", a.eqt), ("PhaseNet", a.pn)):
        models[name] = (load_sbc(path, dev)[1], False)
    out = {"gpu": torch.cuda.get_device_name(0), "batch": 64, "results": {}}
    for name, (m, ours) in models.items():
        m.eval()
        x = torch.randn(64, 6000, 3, device=dev) if ours else torch.randn(64, 3, 6000, device=dev)
        fn = (lambda z, m=m: m(z)) if ours else (lambda z, m=m: m(z))
        n = sum(q.numel() for q in m.parameters())
        row = {"params": n}
        for tag, bf16 in (("fp32", False), ("bf16", True)):
            try:
                tp, mem = bench(fn, x, bf16, a.iters)
                row[tag] = {"traces_per_s": tp, "peak_mib": mem}
            except RuntimeError as e:
                row[tag] = {"error": str(e)[:200]}
            torch.cuda.empty_cache()
        out["results"][name] = row
        print(f"[bench] {name:14s} params={n / 1e6:.3f}M " + " ".join(
            f"{t}: {row[t].get('traces_per_s', float('nan')):.0f} tr/s {row[t].get('peak_mib', float('nan')):.0f} MiB"
            for t in ("fp32", "bf16")), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
