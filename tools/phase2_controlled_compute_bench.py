#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.5 controlled Table V re-bench + continuous-data throughput.

Protocol (recorded in output JSON):
  - GPU idle check (optional warning)
  - batch=64, input 6000×3
  - warmup=20 forward passes
  - timed_runs=5, report median traces/s and peak MiB
  - fp32 and bf16 autocast
  - continuous throughput: streaming windows with 50% overlap on synthetic
    1-hour 3C 100 Hz stream (same model forward), windows/s and hours_stream/hour_wall

  python tools/phase2_controlled_compute_bench.py --output-dir results/phase2/2_5_compute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tools.eval_shift_invariance import load_sbc  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--eqt", type=Path, default=Path("outputs/sb_aug_eqt/last.pt"))
    p.add_argument("--pn", type=Path, default=Path("outputs/sb_aug_phasenet/last.pt"))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--iters-per-run", type=int, default=20)
    p.add_argument("--output-dir", type=Path, default=Path("results/phase2/2_5_compute"))
    return p.parse_args()


def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        util, mem = out.split(",")[0:2]
        return {"gpu_util_pct": float(util), "mem_used_mib": float(mem)}
    except Exception as e:
        return {"error": str(e)}


def bench_model(fn, x, *, bf16: bool, warmup: int, runs: int, iters: int):
    rates, mems = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        for _ in range(runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rates.append(x.size(0) * iters / dt)
            mems.append(torch.cuda.max_memory_allocated() / 2**20)
    return {
        "traces_per_s_runs": rates,
        "traces_per_s_median": float(np.median(rates)),
        "peak_mib_runs": mems,
        "peak_mib_median": float(np.median(mems)),
    }


@torch.no_grad()
def continuous_throughput(model, ours: bool, device, hours_stream=1.0, stride_sec=30.0, window_sec=60.0, bf16=True):
    """Synthetic continuous stream: windows/s and realtime factor."""
    sr = 100
    n_samples = int(hours_stream * 3600 * sr)
    # do not allocate full stream; simulate by random windows
    n_windows = int((hours_stream * 3600 - window_sec) / stride_sec) + 1
    batch = 64
    T = int(window_sec * sr)
    model.eval()
    processed = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        while processed < n_windows:
            b = min(batch, n_windows - processed)
            if ours:
                x = torch.randn(b, T, 3, device=device)
                t = torch.arange(T, device=device).float().unsqueeze(0).expand(b, -1) * (window_sec / T)
                model(x, t)
            else:
                x = torch.randn(b, 3, T, device=device)
                model(x)
            processed += b
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "stream_hours": hours_stream,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "n_windows": n_windows,
        "wall_sec": wall,
        "windows_per_s": n_windows / wall,
        "stream_hours_per_wall_hour": (hours_stream / wall) * 3600.0,
        "bf16": bf16,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    idle = gpu_util()
    protocol = {
        "gpu": torch.cuda.get_device_name(0),
        "batch": args.batch,
        "warmup_forwards": args.warmup,
        "timed_runs": args.runs,
        "iters_per_run": args.iters_per_run,
        "input": "6000x3",
        "idle_snapshot": idle,
        "note": "Median over timed_runs; peak MiB is median of per-run peaks.",
    }
    expected = json.loads((_REPO / "paper/tgrs/expected/compute.json").read_text())
    models = {
        "ours": (load_model(args.ours, device)[0], True),
        "EQTransformer": (load_sbc(args.eqt, device)[1], False),
        "PhaseNet": (load_sbc(args.pn, device)[1], False),
    }
    results = {}
    for name, (m, ours) in models.items():
        m.eval()
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(args.batch, 6000, 3, device=device) if ours else torch.randn(args.batch, 3, 6000, device=device)
        if ours:
            # c2f forward needs time axis — paper_compute_bench used m(z) only; check model
            # CoarseToFine forward(x, t=None) — allow None
            fn = lambda z, m=m: m(z)
        else:
            fn = lambda z, m=m: m(z)
        row = {"params": n, "fp32": {}, "bf16": {}}
        for tag, bf16 in (("fp32", False), ("bf16", True)):
            row[tag] = bench_model(fn, x, bf16=bf16, warmup=args.warmup, runs=args.runs, iters=args.iters_per_run)
        row["continuous_bf16"] = continuous_throughput(m, ours, device)
        results[name] = row
        print(f"[bench] {name} params={n/1e6:.3f}M "
              f"fp32={row['fp32']['traces_per_s_median']:.0f} bf16={row['bf16']['traces_per_s_median']:.0f} "
              f"cont={row['continuous_bf16']['windows_per_s']:.1f} win/s "
              f"({row['continuous_bf16']['stream_hours_per_wall_hour']:.0f}x realtime)", flush=True)
        torch.cuda.empty_cache()

    # diff vs manuscript expected
    diff = {}
    for name in results:
        diff[name] = {}
        exp = expected["results"][name]
        for prec in ("fp32", "bf16"):
            e_tp = exp[prec]["traces_per_s"]
            g_tp = results[name][prec]["traces_per_s_median"]
            e_m = exp[prec]["peak_mib"]
            g_m = results[name][prec]["peak_mib_median"]
            diff[name][prec] = {
                "manuscript_traces_per_s": e_tp,
                "controlled_median_traces_per_s": g_tp,
                "rel_diff_pct": (g_tp - e_tp) / e_tp * 100.0,
                "manuscript_peak_mib": e_m,
                "controlled_median_peak_mib": g_m,
                "peak_mib_diff": g_m - e_m,
            }

    out = {"protocol": protocol, "results": results, "diff_vs_manuscript_table_v": diff}
    (args.output_dir / "controlled_compute.json").write_text(json.dumps(out, indent=2))
    # human summary
    lines = ["# Phase 2.5 controlled compute vs manuscript Table V", ""]
    lines.append(f"GPU: {protocol['gpu']}; batch={protocol['batch']}; warmup={protocol['warmup_forwards']}; runs={protocol['timed_runs']}")
    lines.append("")
    lines.append("| Model | Prec | MS tr/s | Ctrl median | Rel% | MS MiB | Ctrl MiB | Cont win/s | ×realtime |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in results:
        for prec in ("fp32", "bf16"):
            d = diff[name][prec]
            c = results[name]["continuous_bf16"] if prec == "bf16" else None
            cont = f"{results[name]['continuous_bf16']['windows_per_s']:.1f}" if prec == "bf16" else "—"
            rt = f"{results[name]['continuous_bf16']['stream_hours_per_wall_hour']:.0f}" if prec == "bf16" else "—"
            lines.append(
                f"| {name} | {prec} | {d['manuscript_traces_per_s']:.0f} | {d['controlled_median_traces_per_s']:.0f} | "
                f"{d['rel_diff_pct']:+.1f}% | {d['manuscript_peak_mib']:.0f} | {d['controlled_median_peak_mib']:.0f} | {cont} | {rt} |"
            )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
