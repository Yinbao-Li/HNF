#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize interpretable Huygens picking inference on OBS samples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.analyze_stead_picking import load_model
from tools.obs_matched_split import load_split_samples


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain HNF picking on OBS")
    p.add_argument("--checkpoint", default="outputs/run28_obs_full_800/best.pt")
    p.add_argument("--output-dir", default="")
    p.add_argument("--chunks", default="201805,201806,201807,201808")
    p.add_argument("--split-json", default="")
    p.add_argument("--which-split", default="holdout", choices=["train", "holdout"])
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--p-offset-sec", type=float, default=8.0)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_idx(prob: np.ndarray, threshold: float) -> int:
    idx = int(prob.argmax())
    return idx if prob[idx] >= threshold else -1


def _prepare_input(sample: dict, seq_len: int, window_sec: float, normalize_wave) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    x = torch.from_numpy(normalize_wave(sample["wave_3_raw"], "std")).float()
    x = F.interpolate(x.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).squeeze(0)
    x = x.transpose(0, 1).unsqueeze(0)
    t = torch.linspace(0.0, window_sec, seq_len).unsqueeze(-1).unsqueeze(0)
    scale = seq_len / float(sample["wave_3_raw"].shape[-1])
    p_idx = int(max(0, min(seq_len - 1, round(sample["p_idx_native"] * scale))))
    s_idx = -1
    if sample["s_valid"]:
        s_idx = int(max(0, min(seq_len - 1, round(sample["s_idx_native"] * scale))))
    return x, t, p_idx, s_idx


def plot_example(
    out_path: Path,
    sample: dict,
    x: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    kernel_params: dict[str, dict[str, float]],
    pick_threshold: float,
) -> dict:
    x_np = x[0].cpu().numpy()
    t_sec = np.linspace(0.0, 60.0, x_np.shape[0], dtype=np.float32)
    p_prob = torch.sigmoid(outputs["p"][0]).cpu().numpy()
    s_prob = torch.sigmoid(outputs["s"][0]).cpu().numpy()
    rho = outputs["rho"][0].detach().cpu().numpy()
    p_env = outputs["p_envelope"][0].detach().cpu().numpy()
    s_env = outputs["s_envelope"][0].detach().cpu().numpy()
    wave_e = outputs["wave_energy"][0].detach().cpu().numpy()

    p_pred = pick_idx(p_prob, pick_threshold)
    s_pred = pick_idx(s_prob, pick_threshold)
    p_true = int(sample["p_idx_scaled"])
    s_true = int(sample["s_idx_scaled"]) if sample["s_idx_scaled"] >= 0 else -1

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    comp_names = ["Z", "1", "2"]
    for i in range(3):
        axes[0].plot(t_sec, x_np[:, i], lw=0.7, alpha=0.85, label=comp_names[i])
    axes[0].set_ylabel("Waveform")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_title(sample["event_key"])

    axes[1].plot(t_sec, rho, color="saddlebrown", lw=1.0)
    axes[1].set_ylabel("rho(t)")
    axes[1].set_title("Medium density / attenuation proxy")

    det_logit = outputs["det"][0]
    det_scalar = float(det_logit.mean().item()) if det_logit.ndim > 0 else float(det_logit.item())
    det_prob = 1.0 / (1.0 + np.exp(-det_scalar))
    axes[2].plot(t_sec, wave_e, color="gray", lw=0.9, label="shared energy")
    axes[2].axhline(det_prob, color="black", ls="--", lw=1.0, label=f"det prob={det_prob:.2f}")
    axes[2].set_ylabel("Detection")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(t_sec, p_env, color="C0", lw=0.8, alpha=0.6, label="P envelope")
    axes[3].plot(t_sec, p_prob, color="C0", lw=1.2, label="P pick prob")
    axes[3].axvline(t_sec[p_true], color="green", ls="--", lw=1.0, label="P true")
    if p_pred >= 0:
        axes[3].axvline(t_sec[p_pred], color="red", ls=":", lw=1.2, label="P pred")
    axes[3].set_ylabel("P branch")
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].plot(t_sec, s_env, color="C1", lw=0.8, alpha=0.6, label="S envelope")
    axes[4].plot(t_sec, s_prob, color="C1", lw=1.2, label="S pick prob")
    if s_true >= 0:
        axes[4].axvline(t_sec[s_true], color="green", ls="--", lw=1.0, label="S true")
    if s_pred >= 0:
        axes[4].axvline(t_sec[s_pred], color="red", ls=":", lw=1.2, label="S pred")
    axes[4].set_ylabel("S branch")
    axes[4].legend(loc="upper right", fontsize=8)

    if "kernel_contrib" in outputs and p_pred >= 0:
        contrib = outputs["kernel_contrib"][0].detach().cpu().numpy()
        axes[5].plot(t_sec, contrib, color="purple", lw=0.9)
        axes[5].axvline(t_sec[p_pred], color="red", ls=":", lw=1.0)
        axes[5].set_ylabel("|K[p,:]|")
        axes[5].set_title("Huygens causal contributions to P-pick time")
    else:
        axes[5].text(0.5, 0.5, "No kernel row", ha="center", va="center")
        axes[5].set_ylabel("Kernel")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    def _err_sec(pred: int, truth: int) -> float | None:
        if pred < 0 or truth < 0:
            return None
        return float(abs(t_sec[pred] - t_sec[truth]))

    return {
        "event_key": sample["event_key"],
        "chunk": sample["chunk"],
        "p_true_idx": p_true,
        "s_true_idx": s_true,
        "p_pred_idx": p_pred,
        "s_pred_idx": s_pred,
        "p_true_sec": float(t_sec[p_true]),
        "s_true_sec": float(t_sec[s_true]) if s_true >= 0 else None,
        "p_pred_sec": float(t_sec[p_pred]) if p_pred >= 0 else None,
        "s_pred_sec": float(t_sec[s_pred]) if s_pred >= 0 else None,
        "p_abs_err_sec": _err_sec(p_pred, p_true),
        "s_abs_err_sec": _err_sec(s_pred, s_true),
        "kernel_params": kernel_params,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "explain_obs"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_mod = _load_obs_compare_module()
    if args.split_json.strip():
        samples, _, _ = load_split_samples(args.split_json.strip(), args.which_split)
    else:
        chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
        samples, _ = obs_mod.load_obs_windows(
            chunks,
            1000000000,
            args.window_sec,
            args.p_offset_sec,
            args.seed,
            require_full_3c=True,
        )
    if len(samples) == 0:
        raise RuntimeError("No OBS samples available")

    model, _ = load_model(ckpt_path, device, bypass_noise_cancel=False)
    kernel_params = model.collect_kernel_params()
    (out_dir / "kernel_params.json").write_text(json.dumps(kernel_params, indent=2))

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(samples), size=min(args.num_examples, len(samples)), replace=False)
    records = []
    for i, idx in enumerate(indices, start=1):
        s = dict(samples[int(idx)])
        x, t, p_idx, s_idx = _prepare_input(s, args.seq_len, args.window_sec, obs_mod.normalize_wave)
        s["p_idx_scaled"] = p_idx
        s["s_idx_scaled"] = s_idx
        x = x.to(device)
        t = t.to(device)
        with torch.no_grad():
            outputs = model.forward_explain(
                x,
                t,
                include_kernel_row=True,
                kernel_row_idx=p_idx,
                kernel_branch="p",
            )
        meta = plot_example(
            out_dir / f"explain_obs_{i}.png",
            s,
            x.cpu(),
            outputs,
            kernel_params,
            args.pick_threshold,
        )
        records.append(meta)

    p_err = [r["p_abs_err_sec"] for r in records if r["p_abs_err_sec"] is not None]
    s_err = [r["s_abs_err_sec"] for r in records if r["s_abs_err_sec"] is not None]
    summary = {
        "checkpoint": str(ckpt_path),
        "dataset": "OBS",
        "kernel_params": kernel_params,
        "examples": records,
        "aggregate": {
            "n_examples": len(records),
            "mean_p_abs_err_sec": float(np.mean(p_err)) if p_err else None,
            "mean_s_abs_err_sec": float(np.mean(s_err)) if s_err else None,
        },
        "interpretation_notes": {
            "rho(t)": "Learned medium density; higher values indicate stronger attenuation or complexity.",
            "wave_energy": "Shared propagation energy used for event evidence.",
            "p/s_envelope": "P/S branch propagated wave-field envelope before the pick head.",
            "kernel_contrib": "Absolute row of the last P-branch Huygens kernel at the candidate pick time.",
            "obs_note": "OBS often shows stronger site/noise complexity and broader causal support than STEAD.",
        },
    }
    (out_dir / "explain_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[explain-obs] saved {len(records)} figures to {out_dir}")


if __name__ == "__main__":
    main()
