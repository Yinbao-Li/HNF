#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run12 quick ablation for compute optimizations (2-3 epochs each).

Policy:
  - micro-benchmark must beat dense by MIN_SPEEDUP
  - short-train metrics must not regress vs dense ref
  - slow convergence -> drop
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "outputs" / "run12"
STATE_PATH = OUT_ROOT / "state.json"
BASE_RESUME = ROOT / "outputs" / "ablation" / "01_seq800" / "best.pt"

EPOCHS = 3
MIN_SPEEDUP = 1.15
MAX_PS_DROP = 0.03
MIN_SCORE_DELTA = 0.005

QUICK_DATA = {
    "max_event_train": 80000,
    "max_noise_train": 40000,
    "max_val": 10000,
}

COMMON_TAIL = [
    "--seq-len",
    "800",
    "--batch-size",
    "12",
    "--grad-accum-steps",
    "4",
    "--num-workers",
    "0",
    "--embed-dim",
    "64",
    "--num-shared-layers",
    "2",
    "--num-branch-layers",
    "2",
    "--lr",
    "5e-4",
    "--pick-pos-weight",
    "25",
    "--label-sigma-sec",
    "0.4",
    "--local-window-sec",
    "15.0",
    "--seed",
    "42",
    "--max-event-train",
    str(QUICK_DATA["max_event_train"]),
    "--max-noise-train",
    str(QUICK_DATA["max_noise_train"]),
    "--max-val",
    str(QUICK_DATA["max_val"]),
]

EXPERIMENTS = [
    ("12a_dense_ref", {}),
    ("12b_sparse_band", {"sparse_band": True}),
    ("12c_num_anchors128", {"num_anchors": 128}),
]


def build_common(epochs: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools" / "tools/train_stead_picking.py"),
        "--epochs",
        str(epochs),
        *COMMON_TAIL,
    ]


def build_cmd(name: str, flags: dict, epochs: int) -> list[str]:
    cmd = build_common(epochs) + [
        "--output-dir",
        str(OUT_ROOT / name),
        "--resume",
        str(BASE_RESUME),
    ]
    if flags.get("sparse_band"):
        cmd.append("--sparse-band")
    if flags.get("num_anchors"):
        cmd += ["--num-anchors", str(flags["num_anchors"])]
    return cmd


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"completed": [], "results": [], "active": {}}


def save_state(state: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize_history(rows: list[dict]) -> dict:
    if not rows:
        return {}
    last, first = rows[-1], rows[0]
    ep_times = [float(r["ep_time_sec"]) for r in rows if r.get("ep_time_sec")]
    return {
        "epochs": len(rows),
        "score_first": float(first.get("score", 0)),
        "score_last": float(last.get("score", 0)),
        "p_f1_last": float(last.get("p_f1", 0)),
        "s_f1_last": float(last.get("s_f1", 0)),
        "ps_sum_last": float(last.get("p_f1", 0)) + float(last.get("s_f1", 0)),
        "det_f1_last": float(last.get("det_f1", 0)),
        "mean_ep_sec": sum(ep_times) / max(len(ep_times), 1),
    }


@torch.no_grad()
def benchmark_forward(flags: dict, steps: int = 30) -> float:
    from hnf.picking_model import build_picking_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_picking_model(
        embed_dim=64,
        num_shared_layers=2,
        num_branch_layers=2,
        sparse_band=flags.get("sparse_band", False),
        num_anchors=int(flags.get("num_anchors", 0)),
    ).to(device)
    model.eval()
    x = torch.randn(8, 800, 3, device=device)
    t = torch.linspace(0, 60, 800, device=device).view(1, 800, 1).expand(8, -1, -1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        model(x, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def decide(ref: dict, cand: dict, bench_speedup: float) -> tuple[str, str]:
    if bench_speedup < MIN_SPEEDUP:
        return "dropped", f"benchmark {bench_speedup:.2f}x < {MIN_SPEEDUP}"

    ep_speedup = ref.get("mean_ep_sec", 0) / max(cand.get("mean_ep_sec", 1e-6), 1e-6)
    if ep_speedup < MIN_SPEEDUP:
        return "dropped", f"epoch time {ep_speedup:.2f}x < {MIN_SPEEDUP}"

    ps_drop = ref.get("ps_sum_last", 0) - cand.get("ps_sum_last", 0)
    if ps_drop > MAX_PS_DROP:
        return "dropped", f"P+S dropped {ps_drop:.4f} > {MAX_PS_DROP}"

    score_delta = cand.get("score_last", 0) - cand.get("score_first", 0)
    if score_delta < MIN_SCORE_DELTA:
        return "dropped", f"score gain {score_delta:.4f} < {MIN_SCORE_DELTA}"

    return "kept", (
        f"bench={bench_speedup:.2f}x ep={ep_speedup:.2f}x "
        f"ps_sum={cand.get('ps_sum_last', 0):.4f}"
    )


def run_train(name: str, flags: dict, epochs: int) -> int:
    cmd = build_cmd(name, flags, epochs)
    print(f"[run12] >>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run12 compute ablations (short epochs)")
    p.add_argument("--only", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()

    state = load_state()
    runs = EXPERIMENTS
    if args.only:
        runs = [r for r in EXPERIMENTS if r[0] == args.only]
        if not runs:
            raise SystemExit(f"Unknown run: {args.only}")

    bench_dense = None
    if not args.skip_benchmark and not args.dry_run:
        print("[run12] micro-benchmark dense ref ...", flush=True)
        bench_dense = benchmark_forward({})
        state["benchmark_dense_sec"] = bench_dense
        save_state(state)

    ref_summary: dict = {}
    pending_skip: set[str] = set()

    for name, flags in runs:
        if name in state.get("completed", []):
            print(f"[run12] skip completed {name}", flush=True)
            rows = read_history(OUT_ROOT / name / "history.csv")
            summary = summarize_history(rows)
            if name == "12a_dense_ref":
                ref_summary = summary
            continue

        if name != "12a_dense_ref" and bench_dense is not None:
            bench_cand = benchmark_forward(flags)
            speedup = bench_dense / max(bench_cand, 1e-9)
            print(
                f"[run12] benchmark {name}: {bench_cand*1000:.1f}ms speedup={speedup:.2f}x",
                flush=True,
            )
            state.setdefault("benchmarks", {})[name] = {
                "sec": bench_cand,
                "speedup_vs_dense": speedup,
            }
            save_state(state)
            if speedup < MIN_SPEEDUP:
                reason = f"benchmark {speedup:.2f}x < {MIN_SPEEDUP}"
                print(f"[run12] SKIP {name}: {reason}", flush=True)
                state.setdefault("results", []).append(
                    {"name": name, "status": "dropped", "reason": reason, "flags": flags}
                )
                state.setdefault("completed", []).append(name)
                save_state(state)
                continue

        if args.dry_run:
            print(f"[run12] dry-run {name} {flags}", flush=True)
            continue

        if run_train(name, flags, args.epochs) != 0:
            raise SystemExit(1)

        rows = read_history(OUT_ROOT / name / "history.csv")
        summary = summarize_history(rows)
        entry = {"name": name, "flags": flags, **summary}
        state.setdefault("results", []).append(entry)
        state.setdefault("completed", []).append(name)

        if name == "12a_dense_ref":
            ref_summary = summary
        elif ref_summary:
            b_speedup = (state.get("benchmarks", {}).get(name, {}) or {}).get(
                "speedup_vs_dense", 0.0
            )
            status, reason = decide(ref_summary, summary, b_speedup)
            entry["status"] = status
            entry["reason"] = reason
            if status == "kept":
                state["active"] = dict(flags)
            print(f"[run12] {name} -> {status}: {reason}", flush=True)

        save_state(state)

    print("[run12] done", flush=True)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
