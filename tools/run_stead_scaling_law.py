#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate STEAD sample-efficiency / scaling-law sweep.

Trains HNF (run28 recipe), PhaseNet, EQTransformer from scratch at multiple
N_event budgets; evaluates on a shared fixed test subset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


# Absolute event budgets (noise = half). Caps at 30k by default; low-N
# (50/200/500 @ 30ep) uses logs/run_stead_scaling_law_lowN_30ep.sh.
DEFAULT_N_EVENTS = [1000, 3000, 10000, 30000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="outputs/stead_scaling_law")
    p.add_argument("--n-events", default=",".join(str(x) for x in DEFAULT_N_EVENTS))
    p.add_argument("--models", default="hnf,phasenet,eqtransformer")
    p.add_argument("--epochs-hnf", type=int, default=12)
    p.add_argument("--epochs-sb", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-max-events", type=int, default=8000)
    p.add_argument("--eval-max-noise", type=int, default=2000)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def hnf_cmd(out: Path, n_ev: int, n_nz: int, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "tools/train_stead_picking.py",
        "--output-dir",
        str(out),
        "--epochs",
        str(args.epochs_hnf),
        "--batch-size",
        "4",
        "--grad-accum-steps",
        "12",
        "--lr",
        "3e-4",
        "--seq-len",
        "800",
        "--embed-dim",
        "64",
        "--num-shared-layers",
        "2",
        "--num-branch-layers",
        "2",
        "--local-window-sec",
        "15.0",
        "--seed",
        str(args.seed),
        "--max-event-train",
        str(n_ev),
        "--max-noise-train",
        str(n_nz),
        "--max-val",
        "8000",
        "--multi-scale",
        "--principle",
        "huygens_fresnel",
        "--obliquity-scale",
        "1.0",
        "--rho-sparsity-weight",
        "0.02",
        "--rho-sparsity-radius-sec",
        "1.5",
        "--kernel-phys-prior-weight",
        "0.005",
        "--sparse-band",
        "--pick-head-hidden",
        "48",
        "--pick-head-layers",
        "4",
        "--pick-head-kernel",
        "7",
        "--noise-source-dim",
        "16",
        "--no-residual-det-head",
        "--enhanced-det-head",
        "--noise-cancel",
        "--noise-pick-cues",
        "--noise-det-pick-split",
        "--noise-cancel-weight",
        "0.05",
        "--wrong-peak-loss-weight",
        "0.15",
        "--wrong-peak-radius-sec",
        "0.45",
        "--wrong-peak-margin",
        "0.25",
        "--s-wrong-peak-scale",
        "1.35",
        "--ps-order-loss-weight",
        "0.12",
        "--ps-min-gap-sec",
        "0.1",
        "--post-process-p-before-s",
        "--pick-loss-weight",
        "2.8",
        "--pick-pos-weight",
        "28",
        "--p-pick-loss-weight",
        "1.3",
        "--s-pick-loss-weight",
        "1.6",
        "--det-event-weight",
        "2.0",
        "--label-sigma-sec",
        "0.35",
        "--score-mode",
        "det_guard",
        "--det-score-floor",
        "0.988",
        "--device",
        args.device,
    ]


def sb_cmd(model: str, out: Path, n_ev: int, n_nz: int, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "tools/train_seisbench_stead_fraction.py",
        "--model",
        model,
        "--output-dir",
        str(out),
        "--max-event-train",
        str(n_ev),
        "--max-noise-train",
        str(n_nz),
        "--epochs",
        str(args.epochs_sb),
        "--batch-size",
        "8",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--eval-max-events",
        str(args.eval_max_events),
        "--eval-max-noise",
        str(args.eval_max_noise),
    ]


def hnf_subset_eval_cmd(ckpt: Path, out_json: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "tools/eval_stead_scaling_subset.py",
        "--checkpoint",
        str(ckpt),
        "--output-json",
        str(out_json),
        "--max-events",
        str(args.eval_max_events),
        "--max-noise",
        str(args.eval_max_noise),
        "--device",
        args.device,
    ]


def run(cmd: list[str], log: Path, dry: bool) -> int:
    print("[run]", " ".join(cmd), flush=True)
    if dry:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        return subprocess.call(cmd, cwd=str(_REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    n_list = [int(x) for x in args.n_events.split(",") if x.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    manifest = {"n_events": n_list, "models": models, "jobs": []}

    for n_ev in n_list:
        n_nz = max(1, n_ev // 2)
        for model in models:
            tag = f"{model}_N{n_ev}"
            out = root / tag
            metrics_path = out / "test_metrics.json"
            job = {"model": model, "n_event": n_ev, "n_noise": n_nz, "dir": str(out)}
            if args.skip_existing and metrics_path.exists():
                print(f"[skip] {tag}", flush=True)
                job["status"] = "skipped"
                manifest["jobs"].append(job)
                continue
            out.mkdir(parents=True, exist_ok=True)
            if model == "hnf":
                ckpt = out / "best.pt"
                # If a checkpoint already exists (e.g. train finished but subset
                # eval was interrupted), only re-run the shared-subset eval.
                if args.skip_existing and ckpt.exists() and not metrics_path.exists():
                    print(f"[eval-only] {tag} (reuse {ckpt})", flush=True)
                    code = 0
                else:
                    code = run(hnf_cmd(out, n_ev, n_nz, args), out / "train.log", args.dry_run)
                if code == 0 and not args.dry_run:
                    # re-eval on shared subset for fair compare
                    code = run(
                        hnf_subset_eval_cmd(ckpt, metrics_path, args),
                        out / "subset_eval.log",
                        False,
                    )
            elif model in {"phasenet", "eqtransformer"}:
                code = run(sb_cmd(model, out, n_ev, n_nz, args), out / "train.log", args.dry_run)
            else:
                raise ValueError(model)
            job["status"] = "ok" if code == 0 else f"fail:{code}"
            manifest["jobs"].append(job)
            (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    # analyze if any metrics exist
    if not args.dry_run:
        subprocess.call(
            [sys.executable, "tools/analyze_stead_scaling_law.py", "--root", str(root)],
            cwd=str(_REPO_ROOT),
        )


if __name__ == "__main__":
    main()
