#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential ablation: keep strategy only if test P+S F1 improves."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "ablation"
STATE_PATH = OUT_ROOT / "state.json"
LOG_PATH = OUT_ROOT / "ablation.log"

# run7 strict test baseline
BASELINE_PS = 0.8878643517188785 + 0.8873206162507226

BASE_CONFIG = {
    "seq_len": 400,
    "batch_size": 16,
    "grad_accum_steps": 4,
    "epochs": 8,
    "num_workers": 0,
    "embed_dim": 64,
    "num_shared_layers": 2,
    "num_branch_layers": 2,
    "lr": 5e-4,
    "pick_pos_weight": 25,
    "label_sigma_sec": 0.4,
    "local_window_sec": 15.0,
    "seed": 42,
    "focal_gamma": 0.0,
    "s_pick_loss_weight": 1.0,
    "pick_head_hidden": 24,
    "pick_head_kernel": 7,
    "per_time_det": False,
    "post_process_p_before_s": False,
}

ARCH_KEYS = (
    "embed_dim",
    "num_shared_layers",
    "num_branch_layers",
    "per_time_det",
    "pick_head_hidden",
    "pick_head_kernel",
)

STRATEGIES = [
    (
        "01_seq800",
        {"seq_len": 800, "batch_size": 12, "grad_accum_steps": 4},
        "outputs/stead_hnf_picking_run7/best.pt",
    ),
    ("02_per_time_det", {"per_time_det": True}, None),
    ("03_focal_sharp", {"focal_gamma": 2.0, "label_sigma_sec": 0.28}, None),
    (
        "04_deep96",
        {
            "embed_dim": 96,
            "num_shared_layers": 3,
            "num_branch_layers": 3,
        },
        None,
    ),
    ("05_pick_head48", {"pick_head_hidden": 48, "pick_head_kernel": 11}, None),
    ("06_window20", {"local_window_sec": 20.0}, None),
    ("07_s_boost", {"s_pick_loss_weight": 2.5}, None),
]

POST_PROCESS_ONLY = ("08_post_p_before_s", {"post_process_p_before_s": True})


def log(msg: str) -> None:
    print(msg, flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def arch_compatible(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in ARCH_KEYS)


def build_cmd(cfg: dict, out_dir: Path, resume: Optional[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "train_stead_picking.py"),
        "--seq-len",
        str(cfg["seq_len"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--grad-accum-steps",
        str(cfg["grad_accum_steps"]),
        "--epochs",
        str(cfg["epochs"]),
        "--num-workers",
        str(cfg["num_workers"]),
        "--embed-dim",
        str(cfg["embed_dim"]),
        "--num-shared-layers",
        str(cfg["num_shared_layers"]),
        "--num-branch-layers",
        str(cfg["num_branch_layers"]),
        "--lr",
        str(cfg["lr"]),
        "--pick-pos-weight",
        str(cfg["pick_pos_weight"]),
        "--label-sigma-sec",
        str(cfg["label_sigma_sec"]),
        "--local-window-sec",
        str(cfg["local_window_sec"]),
        "--seed",
        str(cfg["seed"]),
        "--pick-head-hidden",
        str(cfg["pick_head_hidden"]),
        "--pick-head-kernel",
        str(cfg["pick_head_kernel"]),
        "--s-pick-loss-weight",
        str(cfg["s_pick_loss_weight"]),
        "--focal-gamma",
        str(cfg["focal_gamma"]),
        "--output-dir",
        str(out_dir),
    ]
    if cfg.get("per_time_det"):
        cmd.append("--per-time-det")
    if cfg.get("post_process_p_before_s"):
        cmd.append("--post-process-p-before-s")
    if resume:
        cmd.extend(["--resume", resume])
    return cmd


def load_test_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def ps_sum(metrics: dict) -> float:
    return float(metrics["p_f1"]) + float(metrics["s_f1"])


def save_state(active: dict, best_ps: float, ckpt: Optional[str], history: list) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "active_config": active,
                "best_ps_sum": best_ps,
                "active_checkpoint": ckpt,
                "history": history,
            },
            indent=2,
        )
    )


def run_train(cfg: dict, name: str, resume_hint: Optional[str], prev_cfg: dict, prev_ckpt: Optional[str]):
    out_dir = OUT_ROOT / name
    resume = None
    if resume_hint and Path(resume_hint).is_file():
        resume = resume_hint
    elif prev_ckpt and arch_compatible(cfg, prev_cfg):
        resume = prev_ckpt

    cmd = build_cmd(cfg, out_dir, resume)
    log(f"[ablation] START {name}  resume={resume or 'scratch'}")
    log("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    metrics = load_test_metrics(out_dir / "test_metrics.json")
    return metrics, str(out_dir / "best.pt")


def eval_post_only(cfg: dict, name: str, ckpt: str) -> dict:
    """Re-evaluate best checkpoint with post-process flag (no retrain)."""
    from eval_stead_picking import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        ckpt,
        cfg,
        post_process_p_before_s=cfg.get("post_process_p_before_s", False),
    )
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text())
        active = state["active_config"]
        best_ps = state["best_ps_sum"]
        active_ckpt = state.get("active_checkpoint")
        history = state.get("history", [])
        log(f"[ablation] resume pipeline  best_ps={best_ps:.4f}")
    else:
        active = dict(BASE_CONFIG)
        best_ps = BASELINE_PS
        active_ckpt = None
        history = []
        log(f"[ablation] baseline run7 test P+S = {best_ps:.4f}")

    for item in STRATEGIES:
        name, delta, resume_hint = item
        if any(h["name"] == name for h in history):
            log(f"[ablation] skip {name} (already done)")
            continue
        trial = {**active, **delta}
        prev_cfg = dict(active)
        try:
            metrics, ckpt = run_train(trial, name, resume_hint, prev_cfg, active_ckpt)
        except subprocess.CalledProcessError as exc:
            log(f"[ablation] FAILED {name}: {exc}")
            history.append({"name": name, "status": "failed", "delta": delta})
            save_state(active, best_ps, active_ckpt, history)
            continue

        cur = ps_sum(metrics)
        kept = cur > best_ps + 1e-4
        record = {
            "name": name,
            "status": "kept" if kept else "dropped",
            "delta": delta,
            "p_f1": metrics["p_f1"],
            "s_f1": metrics["s_f1"],
            "ps_sum": cur,
            "det_f1": metrics["det_f1"],
        }
        history.append(record)
        if kept:
            active = trial
            best_ps = cur
            active_ckpt = ckpt
            log(
                f"[ablation] KEPT {name}  P={metrics['p_f1']:.4f}  "
                f"S={metrics['s_f1']:.4f}  sum={cur:.4f}"
            )
        else:
            log(
                f"[ablation] DROPPED {name}  P={metrics['p_f1']:.4f}  "
                f"S={metrics['s_f1']:.4f}  sum={cur:.4f}  (best={best_ps:.4f})"
            )
        save_state(active, best_ps, active_ckpt, history)

    # Post-process only: no retrain
    pp_name, pp_delta = POST_PROCESS_ONLY
    if active_ckpt and not any(h["name"] == pp_name for h in history):
        trial = {**active, **pp_delta}
        log(f"[ablation] eval-only {pp_name} on {active_ckpt}")
        try:
            metrics = eval_post_only(trial, pp_name, active_ckpt)
            cur = ps_sum(metrics)
            kept = cur > best_ps + 1e-4
            record = {
                "name": pp_name,
                "status": "kept" if kept else "dropped",
                "delta": pp_delta,
                "p_f1": metrics["p_f1"],
                "s_f1": metrics["s_f1"],
                "ps_sum": cur,
                "eval_only": True,
            }
            history.append(record)
            if kept:
                active = trial
                best_ps = cur
                log(f"[ablation] KEPT {pp_name}  sum={cur:.4f}")
            else:
                log(f"[ablation] DROPPED {pp_name}  sum={cur:.4f}")
            save_state(active, best_ps, active_ckpt, history)
        except Exception as exc:
            log(f"[ablation] post-process eval failed: {exc}")

    log(
        f"[ablation] DONE  best P+S={best_ps:.4f}  "
        f"vs baseline {BASELINE_PS:.4f}  delta={best_ps - BASELINE_PS:+.4f}"
    )


if __name__ == "__main__":
    main()
