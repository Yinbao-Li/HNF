#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the run28-style HNF picking model on OBS event windows.

Default: resume from STEAD run28 weights, then fine-tune all layers on OBS.
Random init (--from-scratch) is supported but usually collapses to t=0 picks
without the run28 backbone prior.

Primary validation metric follows the existing OBS paper script:
pick-only P/S F1 on event windows with 0.5 s tolerance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class _CrToNl:
    """Make tqdm progress visible under `tail -f` (tqdm uses \\r by default)."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        self._stream.write(s.replace("\r", "\n"))
        self._stream.flush()

    def flush(self):
        self._stream.flush()


_TQDM_FILE = _CrToNl(sys.stderr)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.picking_model import build_picking_model
from hnf.stead_picking_dataset import gaussian_pick_label
from hnf.noise_cancel import noise_cancel_losses
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import (
    move_batch_to_device,
    ps_order_loss,
    set_seed,
    weighted_pick_loss,
    wrong_peak_rank_loss,
)
from tools.obs_matched_split import load_split_samples

DEFAULT_RUN28_CKPT = "outputs/run28/28_ms_fresnel_phys_20ep/best.pt"


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class ItemDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class LazySampleDataset(Dataset):
    """Build train tensors on the fly to avoid OOM from materializing all items."""

    def __init__(
        self,
        samples: list[dict],
        seq_len: int,
        window_sec: float,
        label_sigma_sec: float,
        normalize_wave,
        *,
        augment: bool,
        seed: int,
        input_dim: int = 3,
    ):
        self.samples = samples
        self.seq_len = int(seq_len)
        self.window_sec = float(window_sec)
        self.sigma = max(1.0, float(label_sigma_sec) * self.seq_len / self.window_sec)
        self.normalize_wave = normalize_wave
        self.augment = bool(augment)
        self.seed = int(seed)
        self.input_dim = int(input_dim)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return _sample_to_item(
            self.samples[idx],
            idx=idx,
            seq_len=self.seq_len,
            window_sec=self.window_sec,
            sigma=self.sigma,
            normalize_wave=self.normalize_wave,
            augment=self.augment,
            seed=self.seed,
            input_dim=self.input_dim,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train run28-style HNF on OBS (run28 init by default)")
    p.add_argument("--output-dir", default="outputs/run28_obs_full_800")
    p.add_argument(
        "--resume",
        default=DEFAULT_RUN28_CKPT,
        help="STEAD run28 checkpoint to initialize from (ignored with --from-scratch)",
    )
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Random init instead of run28 resume (not recommended)",
    )
    p.add_argument("--chunks", default="201805,201806,201807,201808")
    p.add_argument(
        "--split-json",
        default="",
        help="Optional disjoint train/holdout split from tools/obs_matched_split.py",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="0 means use all candidate event windows from the selected OBS chunks",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--p-offset-sec", type=float, default=8.0)
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument(
        "--input-dim",
        type=int,
        default=3,
        choices=[3, 4],
        help="3=Z12 land-compatible; 4=full OBS Z12H including hydrophone",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-threshold", type=float, default=0.25)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--pick-pos-weight", type=float, default=28.0)
    p.add_argument("--pick-loss-weight", type=float, default=2.8)
    p.add_argument("--p-pick-loss-weight", type=float, default=1.3)
    p.add_argument("--s-pick-loss-weight", type=float, default=1.6)
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--noise-pick-penalty", type=float, default=0.0)
    p.add_argument("--wrong-peak-loss-weight", type=float, default=0.15)
    p.add_argument("--wrong-peak-radius-sec", type=float, default=0.45)
    p.add_argument("--wrong-peak-margin", type=float, default=0.25)
    p.add_argument("--s-wrong-peak-scale", type=float, default=1.35)
    p.add_argument("--ps-order-loss-weight", type=float, default=0.12)
    p.add_argument("--ps-min-gap-sec", type=float, default=0.1)
    p.add_argument("--noise-cancel-weight", type=float, default=0.0)
    p.add_argument("--nc-consistency-weight", type=float, default=0.5)
    p.add_argument("--nc-phase-weight", type=float, default=0.1)
    p.add_argument("--nc-preserve-weight", type=float, default=0.3)
    p.add_argument("--nc-energy-weight", type=float, default=0.05)
    p.add_argument("--nc-noise-suppress-weight", type=float, default=0.2)
    p.add_argument("--enable-preserve-gate", action="store_true")
    p.add_argument("--phase-exist", action="store_true", help="Enable P/S existence heads + gate")
    p.add_argument("--phase-exist-hidden", type=int, default=64)
    p.add_argument("--p-late-wrong-peak-weight", type=float, default=0.0,
                   help="Extra wrong-peak penalty for rivals after GT (OBS late bias)")
    p.add_argument("--wrong-peak-listwise-weight", type=float, default=0.0)
    p.add_argument("--wrong-peak-listwise-topk", type=int, default=5)
    p.add_argument(
        "--parallel-exist-fuse",
        action="store_true",
        help="Field-only P/S exist heads parallel to pick + fuse MLP → final pick",
    )
    p.add_argument("--exist-loss-weight", type=float, default=1.0)
    p.add_argument("--s-exist-pos-weight", type=float, default=1.35)
    p.add_argument("--s-exist-neg-weight", type=float, default=1.0)
    p.add_argument("--s-absent-pick-penalty", type=float, default=0.0)
    p.add_argument("--exist-pick-couple-weight", type=float, default=0.0)
    p.add_argument("--soft-gate-s-weight", type=float, default=0.0,
                   help="BCE on sigmoid(s)*sigmoid(s_exist) vs s_target (all windows)")
    p.add_argument(
        "--channel-mask-mode",
        default="strict",
        choices=["strict", "soft_h"],
        help="4C channel filter: strict=all alive; soft_h=require Z12, zero dead H",
    )
    p.add_argument("--exist-th", type=float, default=0.5)
    p.add_argument("--gate-mode", default="hard", choices=["hard", "soft", "soft_floor"])
    p.add_argument("--soft-th", type=float, default=0.25)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--amp", action="store_true")

    # run28-compatible architecture defaults
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-shared-layers", type=int, default=2)
    p.add_argument("--num-branch-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--local-window-sec", type=float, default=15.0)
    p.add_argument("--multi-scale", action="store_true", default=True)
    p.add_argument("--sparse-band", action="store_true", default=True)
    p.add_argument("--principle", choices=["huygens", "huygens_fresnel"], default="huygens_fresnel")
    p.add_argument("--obliquity-scale", type=float, default=1.0)
    p.add_argument("--obliquity-mode", default="none")
    p.add_argument("--obliquity-mix", type=float, default=0.25)
    p.add_argument("--pick-head-hidden", type=int, default=48)
    p.add_argument("--pick-head-kernel", type=int, default=7)
    p.add_argument("--pick-head-layers", type=int, default=4)
    p.add_argument("--noise-cancel", action="store_true", default=True)
    p.add_argument("--noise-source-dim", type=int, default=16)
    p.add_argument("--noise-det-pick-split", action="store_true", default=True)
    p.add_argument("--noise-pick-cues", action="store_true", default=True)
    p.add_argument("--enhanced-det-head", action="store_true", default=True)
    p.add_argument("--residual-pick-head", action="store_true", default=True)
    p.add_argument("--residual-det-head", action="store_true", default=False)
    p.add_argument("--predict-ps-gap", action="store_true", default=False)
    p.add_argument("--ps-gap-hidden", type=int, default=64)
    p.add_argument("--peak-rerank", action="store_true", default=False)
    p.add_argument("--peak-rerank-hidden", type=int, default=16)
    p.add_argument(
        "--p-residual-offset",
        action="store_true",
        default=False,
        help="Post-pick P residual Δt head (waveform crop → bounded offset)",
    )
    p.add_argument("--p-residual-crop-half-bins", type=int, default=30, help="~1.5s at L1200/60s")
    p.add_argument("--p-residual-hidden", type=int, default=32)
    p.add_argument("--p-residual-max-delta-bins", type=float, default=8.0, help="~0.4s cap")
    p.add_argument("--p-residual-loss-weight", type=float, default=0.0)
    p.add_argument(
        "--p-residual-gate-sec",
        type=float,
        default=1.0,
        help="Only supervise residual when coarse peak is within this of GT",
    )
    p.add_argument(
        "--causal-peak-rank",
        action="store_true",
        default=False,
        help="Learned rank head on pick candidates using Huygens field+rho crops",
    )
    p.add_argument("--causal-peak-rank-hidden", type=int, default=48)
    p.add_argument("--causal-peak-rank-topk", type=int, default=8)
    p.add_argument("--causal-peak-rank-crop-half", type=int, default=16)
    p.add_argument("--causal-peak-rank-loss-weight", type=float, default=0.0)
    p.add_argument(
        "--causal-peak-rank-radius-sec",
        type=float,
        default=0.5,
        help="GT must lie within this of a candidate to supervise rank",
    )
    p.add_argument(
        "--train-causal-rank-only",
        action="store_true",
        default=False,
        help="Freeze all params except p_causal_peak_rank",
    )
    p.add_argument(
        "--p-decode-mode",
        default="argmax",
        choices=[
            "argmax",
            "earliest_competitive",
            "score_minus_late",
            "causal_peak_rerank",
            "backbone_causal",
            "learned_causal_rank",
        ],
        help="P pick decode beyond global argmax",
    )
    p.add_argument(
        "--s-decode-mode",
        default="argmax",
        choices=[
            "argmax",
            "earliest_competitive",
            "score_minus_late",
            "causal_peak_rerank",
            "backbone_causal",
            "learned_causal_rank",
        ],
    )
    p.add_argument("--decode-compete-ratio", type=float, default=0.70)
    p.add_argument("--decode-late-penalty", type=float, default=0.15)
    p.add_argument("--decode-echo-gap-lo-sec", type=float, default=1.0)
    p.add_argument("--decode-echo-gap-hi-sec", type=float, default=15.0)
    p.add_argument("--decode-echo-ratio", type=float, default=0.80)
    p.add_argument("--decode-echo-penalty", type=float, default=0.15)
    p.add_argument("--decode-onset-bonus", type=float, default=0.0)
    p.add_argument("--num-anchors", type=int, default=0)
    return p.parse_args()


def shrink_sample_waves(samples: list[dict], input_dim: int) -> None:
    """In-place drop redundant 3C copy (keep float32 — float16 overflows OBS amps)."""
    for s in samples:
        if input_dim >= 4 and s.get("wave_4_raw") is not None:
            s.pop("wave_3_raw", None)
        elif s.get("wave_3_raw") is not None:
            s.pop("wave_4_raw", None)


def _sample_to_item(
    s: dict,
    *,
    idx: int,
    seq_len: int,
    window_sec: float,
    sigma: float,
    normalize_wave,
    augment: bool,
    seed: int,
    input_dim: int,
) -> dict:
    raw = s.get("wave_4_raw") if input_dim >= 4 else None
    if raw is None:
        raw = s["wave_3_raw"]
    wave = np.asarray(raw[:input_dim], dtype=np.float32)
    x = torch.from_numpy(normalize_wave(wave, "std")).float()
    x = F.interpolate(x.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).squeeze(0)
    x = x.transpose(0, 1)
    scale = seq_len / float(wave.shape[-1])
    p_idx = int(max(0, min(seq_len - 1, round(s["p_idx_native"] * scale))))
    if s["s_valid"]:
        s_idx = int(max(0, min(seq_len - 1, round(s["s_idx_native"] * scale))))
        s_valid = 1.0
    else:
        s_idx = 0
        s_valid = 0.0
    if augment:
        rng = np.random.default_rng(seed + 9973 * (idx + 1))
        amp = float(rng.uniform(0.75, 1.35))
        noise_scale = float(rng.uniform(0.01, 0.08))
        shift = int(rng.integers(-max(1, seq_len // 120), max(2, seq_len // 120 + 1)))
        x = x * amp + torch.randn_like(x) * noise_scale
        x = torch.roll(x, shifts=shift, dims=0)
        p_idx = int(max(0, min(seq_len - 1, p_idx + shift)))
        if s_valid > 0.5:
            s_idx = int(max(0, min(seq_len - 1, s_idx + shift)))
    return {
        "x": x,
        "t": torch.linspace(0.0, window_sec, seq_len).unsqueeze(-1),
        "det": torch.tensor(1.0, dtype=torch.float32),
        "p_idx": torch.tensor(p_idx, dtype=torch.long),
        "s_idx": torch.tensor(s_idx, dtype=torch.long),
        "p_valid": torch.tensor(1.0, dtype=torch.float32),
        "s_valid": torch.tensor(s_valid, dtype=torch.float32),
        "p_target": gaussian_pick_label(p_idx, seq_len, sigma),
        "s_target": gaussian_pick_label(s_idx, seq_len, sigma)
        if s_valid > 0.5
        else torch.zeros(seq_len, dtype=torch.float32),
        "event_key": s["event_key"],
    }


def build_items(
    samples: list[dict],
    seq_len: int,
    window_sec: float,
    label_sigma_sec: float,
    normalize_wave,
    *,
    augment: bool,
    seed: int,
    input_dim: int = 3,
) -> list[dict]:
    sigma = max(1.0, label_sigma_sec * seq_len / window_sec)
    input_dim = int(input_dim)
    items = []
    for i, s in enumerate(
        tqdm(
            samples,
            desc="build_items",
            leave=True,
            file=_TQDM_FILE,
            ncols=100,
            mininterval=2.0,
            disable=False,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    ):
        items.append(
            _sample_to_item(
                s,
                idx=i,
                seq_len=seq_len,
                window_sec=window_sec,
                sigma=sigma,
                normalize_wave=normalize_wave,
                augment=augment,
                seed=seed,
                input_dim=input_dim,
            )
        )
        # Drop raw waves after materializing to cut peak RSS (OOM on soft_h 47k).
        s.pop("wave_4_raw", None)
        s.pop("wave_3_raw", None)
    return items


def collate(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "det": torch.stack([b["det"] for b in batch]),
        "p_idx": torch.stack([b["p_idx"] for b in batch]),
        "s_idx": torch.stack([b["s_idx"] for b in batch]),
        "p_valid": torch.stack([b["p_valid"] for b in batch]),
        "s_valid": torch.stack([b["s_valid"] for b in batch]),
        "p_target": torch.stack([b["p_target"] for b in batch]),
        "s_target": torch.stack([b["s_target"] for b in batch]),
    }


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_picking_model(
        input_dim=int(args.input_dim),
        embed_dim=args.embed_dim,
        num_shared_layers=args.num_shared_layers,
        num_branch_layers=args.num_branch_layers,
        local_window_sec=args.local_window_sec,
        dropout=args.dropout,
        pick_head_hidden=args.pick_head_hidden,
        pick_head_kernel=args.pick_head_kernel,
        pick_head_layers=args.pick_head_layers,
        multi_scale=args.multi_scale,
        sparse_band=args.sparse_band,
        num_anchors=args.num_anchors,
        residual_pick_head=args.residual_pick_head,
        residual_det_head=args.residual_det_head,
        enhanced_det_head=args.enhanced_det_head,
        noise_cancel=args.noise_cancel,
        noise_source_dim=args.noise_source_dim,
        noise_det_pick_split=args.noise_det_pick_split,
        noise_pick_cues=args.noise_pick_cues,
        principle=args.principle,
        obliquity_scale=args.obliquity_scale,
        obliquity_mode=args.obliquity_mode,
        obliquity_mix=args.obliquity_mix,
        predict_ps_gap=args.predict_ps_gap,
        ps_gap_hidden=args.ps_gap_hidden,
        peak_rerank=args.peak_rerank,
        peak_rerank_hidden=args.peak_rerank_hidden,
        p_residual_offset=bool(getattr(args, "p_residual_offset", False)),
        p_residual_crop_half_bins=int(getattr(args, "p_residual_crop_half_bins", 30)),
        p_residual_hidden=int(getattr(args, "p_residual_hidden", 32)),
        p_residual_max_delta_bins=float(getattr(args, "p_residual_max_delta_bins", 8.0)),
        causal_peak_rank=bool(getattr(args, "causal_peak_rank", False)),
        causal_peak_rank_hidden=int(getattr(args, "causal_peak_rank_hidden", 48)),
        causal_peak_rank_topk=int(getattr(args, "causal_peak_rank_topk", 8)),
        causal_peak_rank_crop_half=int(getattr(args, "causal_peak_rank_crop_half", 16)),
        phase_exist=bool(getattr(args, "phase_exist", False)),
        phase_exist_hidden=int(getattr(args, "phase_exist_hidden", 64)),
        parallel_exist_fuse=bool(getattr(args, "parallel_exist_fuse", False)),
    ).to(device)
    if args.enable_preserve_gate and getattr(model, "noise_cancel_branch", None) is not None:
        model.noise_cancel_branch.enable_preserve_gate = True
    return model


def ensure_optional_heads(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    """Attach CLI-requested heads missing from a resumed checkpoint (random init)."""
    from hnf.picking_model import LocalPeakRerankHead, PPeakResidualOffset

    attached: list[str] = []
    if bool(getattr(args, "peak_rerank", False)) and getattr(model, "p_peak_rerank", None) is None:
        hid = int(getattr(args, "peak_rerank_hidden", 16))
        model.p_peak_rerank = LocalPeakRerankHead(hidden=hid).to(device)
        model.s_peak_rerank = LocalPeakRerankHead(hidden=hid).to(device)
        model.peak_rerank = True
        attached.append("peak_rerank")
    if bool(getattr(args, "p_residual_offset", False)) and getattr(model, "p_residual_offset", None) is None:
        model.p_residual_offset = PPeakResidualOffset(
            input_dim=int(getattr(model, "input_dim", args.input_dim)),
            crop_half_bins=int(getattr(args, "p_residual_crop_half_bins", 30)),
            hidden=int(getattr(args, "p_residual_hidden", 32)),
            max_delta_bins=float(getattr(args, "p_residual_max_delta_bins", 8.0)),
        ).to(device)
        model.p_residual_offset_enabled = True
        attached.append("p_residual_offset")
    if bool(getattr(args, "causal_peak_rank", False)) and getattr(model, "p_causal_peak_rank", None) is None:
        from hnf.causal_peak_rank import CausalPeakRankHead

        model.p_causal_peak_rank = CausalPeakRankHead(
            crop_half_bins=int(getattr(args, "causal_peak_rank_crop_half", 16)),
            hidden=int(getattr(args, "causal_peak_rank_hidden", 48)),
            topk=int(getattr(args, "causal_peak_rank_topk", 8)),
            dropout=float(getattr(args, "dropout", 0.1)),
        ).to(device)
        model.causal_peak_rank_enabled = True
        attached.append("p_causal_peak_rank")
    return attached


def freeze_for_causal_rank_only(model: torch.nn.Module) -> None:
    """Train only the learned causal peak-rank head."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("p_causal_peak_rank.")


def _channel_alive_np(wave: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(np.asarray(wave, dtype=np.float64) ** 2, axis=-1))
    return rms > eps


def filter_alive_channels(
    samples: list[dict],
    input_dim: int,
    *,
    mode: str = "strict",
) -> list[dict]:
    """Filter / repair windows by channel health.

    mode:
      - strict: all input_dim channels must be alive (legacy 4C hydro filter)
      - soft_h: require first 3 alive; zero any dead channel (esp. H) in a copy
    """
    mode = str(mode or "strict")
    keep = []
    for s in samples:
        raw = s.get("wave_4_raw") if input_dim >= 4 else s.get("wave_3_raw")
        if raw is None:
            raw = s["wave_3_raw"]
        wave = np.asarray(raw[:input_dim], dtype=np.float32)
        alive = _channel_alive_np(wave)
        if mode == "soft_h" and input_dim >= 4:
            if not bool(alive[:3].all()):
                continue
            if not bool(alive.all()):
                fixed = dict(s)
                w4 = np.array(s.get("wave_4_raw", wave), dtype=np.float32, copy=True)
                dead = ~_channel_alive_np(w4[:input_dim])
                for ci in np.where(dead)[0]:
                    w4[int(ci)] = 0.0
                fixed["wave_4_raw"] = w4
                fixed["wave_3_raw"] = w4[:3].copy()
                fixed["h_channel_zeroed"] = True
                keep.append(fixed)
            else:
                keep.append(s)
            continue
        if bool(alive.all()):
            keep.append(s)
    return keep


def weighted_phase_exist_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """Per-sample BCE with separate positive/negative class weights."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    w = torch.where(targets > 0.5, float(pos_weight), float(neg_weight))
    return (bce * w).mean()


def gate_supervision_loss(
    out: dict,
    batch: dict,
    args: argparse.Namespace,
    s_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align pick curves with phase absence: suppress S peaks / exist on no-S windows."""
    absent = batch["s_valid"] <= 0.5
    if not absent.any():
        z = s_logits.new_zeros(())
        return z, {}
    s_probs = torch.sigmoid(s_logits)
    s_peak = s_probs.amax(dim=-1)
    parts: dict[str, float] = {}
    loss = s_logits.new_zeros(())
    absent_w = float(getattr(args, "s_absent_pick_penalty", 0.0))
    if absent_w > 0:
        loss_absent_pick = s_peak[absent].mean()
        loss = loss + absent_w * loss_absent_pick
        parts["loss_s_absent_pick"] = float(loss_absent_pick.detach())
    couple_w = float(getattr(args, "exist_pick_couple_weight", 0.0))
    if couple_w > 0 and "s_exist" in out:
        s_exist_prob = torch.sigmoid(out["s_exist"])
        coupled = (s_exist_prob[absent] * s_peak[absent]).mean()
        loss = loss + couple_w * coupled
        parts["loss_exist_pick_couple"] = float(coupled.detach())
    return loss, parts


def compute_loss(
    out: dict,
    batch: dict,
    args: argparse.Namespace,
    model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    p_logits = torch.nan_to_num(out.get("p_logits", out["p"]), nan=-50.0, posinf=50.0, neginf=-50.0)
    s_logits = torch.nan_to_num(out.get("s_logits", out["s"]), nan=-50.0, posinf=50.0, neginf=-50.0)
    p_valid = batch["p_valid"] > 0.5

    # Rank-only FT: never build the frozen-backbone pick loss (no grad_fn).
    if bool(getattr(args, "train_causal_rank_only", False)):
        rank_head = None if model is None else getattr(model, "p_causal_peak_rank", None)
        rank_w = float(getattr(args, "causal_peak_rank_loss_weight", 1.0))
        z = p_logits.new_zeros(())
        stats = {
            "loss": 0.0,
            "loss_p": 0.0,
            "loss_s": 0.0,
            "noise_pen": 0.0,
            "wrong_p": 0.0,
            "wrong_s": 0.0,
            "wrong_list": 0.0,
            "ps_order": 0.0,
            "loss_nc": 0.0,
            "loss_exist": 0.0,
            "loss_gate_sup": 0.0,
            "loss_soft_gate_s": 0.0,
            "loss_p_res": 0.0,
            "loss_rank": 0.0,
        }
        if rank_head is None or "p_field_env" not in out or "rho" not in out or rank_w <= 0:
            # Still attach a dummy param loss so AMP/backward never sees a bare constant.
            if rank_head is not None:
                loss = sum(p.sum() * 0.0 for p in rank_head.parameters())
                stats["loss"] = float(loss.detach())
                return loss, stats
            return z, stats
        from hnf.causal_peak_rank import causal_peak_rank_loss, find_local_peak_candidates

        probs = torch.sigmoid(p_logits).detach()
        field = out["p_field_env"].detach()
        rho = out["rho"].detach()
        peak_idx, _peak_sc, peak_mask = find_local_peak_candidates(
            probs,
            pick_th=float(getattr(args, "pick_threshold", 0.25)),
            topk=int(getattr(rank_head, "topk", 8)),
            compete_ratio=0.0,
        )
        rank_logits = rank_head(probs, field, rho, peak_idx, peak_mask)
        bins_per_sec = float(args.seq_len) / float(getattr(args, "window_sec", 60.0))
        radius_bins = max(
            1,
            int(round(float(getattr(args, "causal_peak_rank_radius_sec", 0.5)) * bins_per_sec)),
        )
        loss_rank = causal_peak_rank_loss(
            rank_logits,
            peak_idx,
            peak_mask,
            batch["p_idx"],
            p_valid.float(),
            radius_bins=radius_bins,
        )
        loss = rank_w * loss_rank
        stats["loss"] = float(loss.detach())
        stats["loss_rank"] = float(loss_rank.detach())
        return loss, stats

    loss_p = weighted_pick_loss(
        p_logits,
        batch["p_target"],
        args.pick_pos_weight,
        focal_gamma=args.focal_gamma,
    )
    s_mask = batch["s_valid"] > 0.5
    if s_mask.any():
        loss_s = weighted_pick_loss(
            s_logits[s_mask],
            batch["s_target"][s_mask],
            args.pick_pos_weight,
            focal_gamma=args.focal_gamma,
        )
    else:
        loss_s = p_logits.new_zeros(())
    noise_pen = p_logits.new_zeros(())
    if args.noise_pick_penalty > 0:
        noise_pen = (torch.sigmoid(p_logits).mean() + torch.sigmoid(s_logits).mean()) * 0.5
    event_mask = batch["det"] > 0.5
    p_valid = batch["p_valid"] > 0.5
    s_valid = batch["s_valid"] > 0.5
    wrong_p = wrong_peak_rank_loss(
        p_logits,
        batch["p_idx"],
        p_valid,
        args.seq_len,
        radius_sec=args.wrong_peak_radius_sec,
        margin=args.wrong_peak_margin,
        late_weight=float(getattr(args, "p_late_wrong_peak_weight", 0.0)),
    )
    wrong_s = wrong_peak_rank_loss(
        s_logits,
        batch["s_idx"],
        s_valid,
        args.seq_len,
        radius_sec=args.wrong_peak_radius_sec,
        margin=args.wrong_peak_margin,
    )
    listwise_w = float(getattr(args, "wrong_peak_listwise_weight", 0.0))
    wrong_list = p_logits.new_zeros(())
    if listwise_w > 0:
        from tools.train_stead_picking import wrong_peak_listwise_loss

        wrong_list = wrong_peak_listwise_loss(
            p_logits,
            batch["p_idx"],
            p_valid,
            args.seq_len,
            radius_sec=args.wrong_peak_radius_sec,
            topk=int(getattr(args, "wrong_peak_listwise_topk", 5)),
        )
    order = ps_order_loss(
        p_logits,
        s_logits,
        event_mask,
        args.seq_len,
        args.ps_min_gap_sec,
    )
    loss_nc = p_logits.new_zeros(())
    nc_parts: dict[str, float] = {}
    if args.noise_cancel_weight > 0 and "nc_u_final" in out:
        nc_out = {"u_final": out["nc_u_final"], "n_sim": out["nc_n_sim"]}
        loss_nc, nc_parts = noise_cancel_losses(
            out,
            nc_out,
            batch,
            consistency_weight=args.nc_consistency_weight,
            phase_weight=args.nc_phase_weight,
            preserve_weight=args.nc_preserve_weight,
            energy_weight=args.nc_energy_weight,
            noise_suppress_weight=args.nc_noise_suppress_weight,
        )
    loss_exist = p_logits.new_zeros(())
    exist_parts: dict[str, float] = {}
    loss_gate_sup = p_logits.new_zeros(())
    gate_parts: dict[str, float] = {}
    if getattr(args, "exist_loss_weight", 0) > 0 and "p_exist" in out and "s_exist" in out:
        loss_exist_p = F.binary_cross_entropy_with_logits(out["p_exist"], batch["p_valid"])
        loss_exist_s = weighted_phase_exist_loss(
            out["s_exist"],
            batch["s_valid"],
            pos_weight=float(getattr(args, "s_exist_pos_weight", 1.0)),
            neg_weight=float(getattr(args, "s_exist_neg_weight", 1.0)),
        )
        loss_exist = loss_exist_p + loss_exist_s
        # Parallel fuse: also supervise refined exist from (field exist + pick summary).
        if "p_exist_ref" in out and "s_exist_ref" in out:
            loss_exist = loss_exist + 0.5 * (
                F.binary_cross_entropy_with_logits(out["p_exist_ref"], batch["p_valid"])
                + weighted_phase_exist_loss(
                    out["s_exist_ref"],
                    batch["s_valid"],
                    pos_weight=float(getattr(args, "s_exist_pos_weight", 1.0)),
                    neg_weight=float(getattr(args, "s_exist_neg_weight", 1.0)),
                )
            )
        exist_parts = {
            "loss_exist_p": float(loss_exist_p.detach()),
            "loss_exist_s": float(loss_exist_s.detach()),
            "loss_exist": float(loss_exist.detach()),
        }
        loss_gate_sup, gate_parts = gate_supervision_loss(out, batch, args, s_logits)
    soft_w = float(getattr(args, "soft_gate_s_weight", 0.0))
    loss_soft_s = p_logits.new_zeros(())
    if soft_w > 0 and "s_exist" in out:
        # Differentiable soft gate: align gated curve to label (0 on no-S windows).
        gated = torch.sigmoid(s_logits) * torch.sigmoid(out["s_exist"]).unsqueeze(-1)
        # clamp for numerical stability in BCE
        gated = gated.clamp(1e-6, 1.0 - 1e-6)
        loss_soft_s = F.binary_cross_entropy(gated, batch["s_target"])
        exist_parts["loss_soft_gate_s"] = float(loss_soft_s.detach())
    loss = (
        args.pick_loss_weight
        * (args.p_pick_loss_weight * loss_p + args.s_pick_loss_weight * loss_s)
        + args.noise_pick_penalty * noise_pen
        + args.wrong_peak_loss_weight * (wrong_p + args.s_wrong_peak_scale * wrong_s)
        + listwise_w * wrong_list
        + args.ps_order_loss_weight * order
        + args.noise_cancel_weight * loss_nc
        + float(getattr(args, "exist_loss_weight", 0.0)) * loss_exist
        +         loss_gate_sup
        + soft_w * loss_soft_s
    )
    loss_p_res = p_logits.new_zeros(())
    res_w = float(getattr(args, "p_residual_loss_weight", 0.0))
    if res_w > 0 and "p_delta_bins" in out and "p_coarse_idx" in out:
        coarse = out["p_coarse_idx"]
        gt = batch["p_idx"]
        bins_per_sec = float(args.seq_len) / float(getattr(args, "window_sec", 60.0))
        gate_bins = max(1, int(round(float(getattr(args, "p_residual_gate_sec", 1.0)) * bins_per_sec)))
        max_d = float(getattr(args, "p_residual_max_delta_bins", 8.0))
        mask = p_valid & ((gt - coarse).abs() <= gate_bins)
        if mask.any():
            target = (gt - coarse).float().clamp(-max_d, max_d)
            loss_p_res = F.smooth_l1_loss(out["p_delta_bins"][mask], target[mask])
            loss = loss + res_w * loss_p_res
    loss_rank = p_logits.new_zeros(())
    rank_w = float(getattr(args, "causal_peak_rank_loss_weight", 0.0))
    rank_head = None if model is None else getattr(model, "p_causal_peak_rank", None)
    if rank_w > 0 and rank_head is not None and "p_field_env" in out and "rho" in out:
        from hnf.causal_peak_rank import causal_peak_rank_loss, find_local_peak_candidates

        probs = torch.sigmoid(p_logits)
        field = out["p_field_env"]
        rho = out["rho"]
        if bool(getattr(args, "train_causal_rank_only", False)):
            probs = probs.detach()
            field = field.detach()
            rho = rho.detach()
        peak_idx, _peak_sc, peak_mask = find_local_peak_candidates(
            probs,
            pick_th=float(getattr(args, "pick_threshold", 0.25)),
            topk=int(getattr(rank_head, "topk", 8)),
            compete_ratio=0.0,
        )
        rank_logits = rank_head(probs, field, rho, peak_idx, peak_mask)
        bins_per_sec = float(args.seq_len) / float(getattr(args, "window_sec", 60.0))
        radius_bins = max(
            1,
            int(round(float(getattr(args, "causal_peak_rank_radius_sec", 0.5)) * bins_per_sec)),
        )
        loss_rank = causal_peak_rank_loss(
            rank_logits,
            peak_idx,
            peak_mask,
            batch["p_idx"],
            p_valid.float(),
            radius_bins=radius_bins,
        )
        if bool(getattr(args, "train_causal_rank_only", False)):
            loss = rank_w * loss_rank
        else:
            loss = loss + rank_w * loss_rank
    stats = {
        "loss": float(loss.detach()),
        "loss_p": float(loss_p.detach()),
        "loss_s": float(loss_s.detach()),
        "noise_pen": float(noise_pen.detach()),
        "wrong_p": float(wrong_p.detach()),
        "wrong_s": float(wrong_s.detach()),
        "wrong_list": float(wrong_list.detach()),
        "ps_order": float(order.detach()),
        "loss_nc": float(loss_nc.detach()),
        "loss_exist": float(loss_exist.detach()),
        "loss_gate_sup": float(loss_gate_sup.detach()),
        "loss_soft_gate_s": float(loss_soft_s.detach()),
        "loss_p_res": float(loss_p_res.detach()),
        "loss_rank": float(loss_rank.detach()),
    }
    stats.update(nc_parts)
    stats.update(exist_parts)
    stats.update(gate_parts)
    return loss, stats


@torch.no_grad()
def eval_pick_diagnostics(
    model,
    samples: list[dict],
    device: torch.device,
    args: argparse.Namespace,
    obs_mod,
    max_n: int = 128,
) -> dict[str, float]:
    """Quick sanity check: is argmax near GT or stuck at t=0?"""
    model.eval()
    tol = max(1, int(round(args.tol_sec * args.seq_len / 60.0)))
    errs: list[int] = []
    pred_zero = 0
    n = 0
    for start in range(0, min(len(samples), max_n), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        x, t, p_idx, _s_idx, _p_valid, _s_valid = obs_mod.to_hnf_batch(
            chunk, args.seq_len, args.window_sec, device, n_channels=int(args.input_dim)
        )
        out = model(x, t)
        pred = torch.sigmoid(torch.nan_to_num(out.get("p_logits", out["p"]))).argmax(dim=-1)
        for i in range(len(chunk)):
            pi = int(pred[i].item())
            gi = int(p_idx[i].item())
            errs.append(abs(pi - gi))
            pred_zero += int(pi == 0)
            n += 1
    if n == 0:
        return {}
    arr = np.asarray(errs, dtype=np.float32)
    return {
        "p_err_mean_bins": float(arr.mean()),
        "p_within_tol": float((arr <= tol).mean()),
        "pred_at_zero_rate": float(pred_zero / n),
    }


@torch.no_grad()
def eval_pick_only(model, samples, device, args, obs_mod) -> dict:
    model.eval()
    use_exist = bool(getattr(args, "phase_exist", False))
    result = obs_mod.eval_hnf(
        model,
        samples,
        device,
        args.seq_len,
        args.window_sec,
        args.pick_threshold,
        args.det_threshold,
        args.tol_sec,
        args.batch_size,
        n_channels=int(getattr(args, "input_dim", 3)),
        exist_th=float(getattr(args, "exist_th", 0.5)),
        score_absent=use_exist,
        gate_mode=str(getattr(args, "gate_mode", "hard")),
        soft_th=float(getattr(args, "soft_th", 0.25)),
        p_decode_mode=str(getattr(args, "p_decode_mode", "argmax")),
        s_decode_mode=str(getattr(args, "s_decode_mode", "argmax")),
        decode_compete_ratio=float(getattr(args, "decode_compete_ratio", 0.70)),
        decode_late_penalty=float(getattr(args, "decode_late_penalty", 0.0)),
        apply_p_residual=bool(getattr(model, "p_residual_offset", None) is not None),
        apply_causal_peak_rank=bool(getattr(model, "p_causal_peak_rank", None) is not None)
        or str(getattr(args, "p_decode_mode", "")) == "learned_causal_rank",
        decode_echo_gap_lo_bins=max(
            1,
            int(
                round(
                    float(getattr(args, "decode_echo_gap_lo_sec", 1.0))
                    * float(args.seq_len)
                    / float(getattr(args, "window_sec", 60.0))
                )
            ),
        ),
        decode_echo_gap_hi_bins=max(
            1,
            int(
                round(
                    float(getattr(args, "decode_echo_gap_hi_sec", 15.0))
                    * float(args.seq_len)
                    / float(getattr(args, "window_sec", 60.0))
                )
            ),
        ),
        decode_echo_ratio=float(getattr(args, "decode_echo_ratio", 0.70)),
        decode_echo_penalty=float(getattr(args, "decode_echo_penalty", 0.35)),
        decode_onset_bonus=float(getattr(args, "decode_onset_bonus", 0.10)),
    )
    metrics = dict(result["pick_only"])
    if "exist_acc" in result:
        metrics["exist_acc"] = result["exist_acc"]
    return metrics


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch: int,
    best_metrics: dict | None,
    history: list[dict],
    extra: dict | None = None,
) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "best_val_pick_only": best_metrics,
        "history": history,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    obs_mod = _load_obs_compare_module()
    holdout_samples: list[dict] = []
    if args.split_json.strip():
        train_pool, load_info, split_meta = load_split_samples(args.split_json.strip(), "train")
        # Defer holdout load until final eval to cut peak RSS on soft_h.
        holdout_samples = []
        hold_info = {"deferred": True}
        load_info = {
            **load_info,
            "holdout_info": hold_info,
            "split_json": args.split_json.strip(),
            "protocol": split_meta.get("protocol"),
            "p_offset_min": split_meta.get("p_offset_min"),
            "p_offset_max": split_meta.get("p_offset_max"),
        }
        # Prefer explicit val_entries (full native split); else carve from train.
        if split_meta.get("val_entries"):
            from tools.obs_matched_split import load_split

            meta = load_split(args.split_json.strip())
            val_samples, val_info = obs_mod.load_obs_windows_from_entries(
                meta["val_entries"], float(meta["window_sec"]), require_full_3c=True
            )
            load_info["val_info"] = val_info
            train_samples = train_pool
        else:
            rng = np.random.default_rng(args.seed)
            idxs = np.arange(len(train_pool))
            rng.shuffle(idxs)
            n_val = max(64, int(round(len(train_pool) * args.val_frac)))
            val_idx = set(idxs[:n_val].tolist())
            val_samples = [train_pool[i] for i in idxs if i in val_idx]
            train_samples = [train_pool[i] for i in idxs if i not in val_idx]
        eval_pool = []  # filled from deferred holdout at end
    else:
        chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
        max_events = None if args.max_events <= 0 else int(args.max_events)
        request_n = max_events if max_events is not None else 1_000_000_000
        samples, load_info = obs_mod.load_obs_windows(
            chunks,
            request_n,
            args.window_sec,
            args.p_offset_sec,
            args.seed,
            require_full_3c=True,
        )
        if max_events is None:
            train_samples = samples
        else:
            train_samples = samples[:max_events]
        rng = np.random.default_rng(args.seed)
        idxs = np.arange(len(train_samples))
        rng.shuffle(idxs)
        n_val = max(64, int(round(len(train_samples) * args.val_frac)))
        val_idx = set(idxs[:n_val].tolist())
        val_samples = [train_samples[i] for i in idxs if i in val_idx]
        train_samples = [train_samples[i] for i in idxs if i not in val_idx]
        eval_pool = val_samples
    if int(args.input_dim) >= 4:
        before = (len(train_samples), len(val_samples), len(holdout_samples), len(eval_pool))
        mask_mode = str(getattr(args, "channel_mask_mode", "strict"))
        # Fair holdout/board stays strict 4C; train/val may soft-mask dead H to recover data.
        train_mode = mask_mode
        eval_mode = "strict"
        train_samples = filter_alive_channels(train_samples, int(args.input_dim), mode=train_mode)
        val_samples = filter_alive_channels(val_samples, int(args.input_dim), mode=train_mode)
        holdout_samples = filter_alive_channels(holdout_samples, int(args.input_dim), mode=eval_mode)
        eval_pool = filter_alive_channels(eval_pool, int(args.input_dim), mode=eval_mode)
        print(
            f"[obs-full] 4C channel filter mode={mask_mode} "
            f"train {before[0]}->{len(train_samples)} "
            f"val {before[1]}->{len(val_samples)} "
            f"holdout {before[2]}->{len(holdout_samples)} (holdout always strict)",
            flush=True,
        )
    shrink_sample_waves(train_samples, int(args.input_dim))
    shrink_sample_waves(val_samples, int(args.input_dim))
    shrink_sample_waves(holdout_samples, int(args.input_dim))
    if eval_pool is not holdout_samples and eval_pool is not val_samples:
        shrink_sample_waves(eval_pool, int(args.input_dim))
    import gc

    gc.collect()
    if len(train_samples) < 100:
        raise RuntimeError(f"Too few OBS training samples: {len(train_samples)}")

    print(
        f"[obs-full] train={len(train_samples)} val={len(val_samples)} "
        f"holdout={len(holdout_samples)} (deferred) device={device} seq_len={args.seq_len} "
        f"input_dim={args.input_dim} nc_w={args.noise_cancel_weight}",
        flush=True,
    )
    print(
        f"[obs-full] building train items ({len(train_samples)}; "
        "frees raw waves as it goes — progress bar next)...",
        flush=True,
    )
    train_items = build_items(
        train_samples,
        args.seq_len,
        args.window_sec,
        args.label_sigma_sec,
        obs_mod.normalize_wave,
        augment=args.augment,
        seed=args.seed,
        input_dim=int(args.input_dim),
    )
    # Drop emptied sample shells; keep only tensors.
    train_samples_n = len(train_samples)
    del train_samples
    train_pool = None  # noqa: F841 — drop load refs after materialize
    import gc

    gc.collect()
    print(f"[obs-full] built {len(train_items)} train items", flush=True)
    train_loader = DataLoader(
        ItemDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )

    ckpt_args: dict = {}
    if args.from_scratch:
        model = build_model(args, device)
        init_tag = "random"
    else:
        resume_path = Path(args.resume or DEFAULT_RUN28_CKPT)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model, ckpt_args = load_model(resume_path, device, bypass_noise_cancel=False)
        attached = ensure_optional_heads(model, args, device)
        if attached:
            print(f"[obs-full] attached new heads: {attached}", flush=True)
        if bool(getattr(args, "train_causal_rank_only", False)):
            freeze_for_causal_rank_only(model)
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[obs-full] train_causal_rank_only: trainable params={n_train}", flush=True)
        model.train()
        init_tag = str(resume_path)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    print(
        f"[obs-full] baseline eval on val subset "
        f"({min(512, len(val_samples))}/{len(val_samples)}; no train bar yet)...",
        flush=True,
    )
    baseline_pool = val_samples[: min(512, len(val_samples))]
    baseline = eval_pick_only(model, baseline_pool, device, args, obs_mod)
    baseline_diag = eval_pick_diagnostics(model, baseline_pool, device, args, obs_mod)
    print(
        f"[obs-full] init={init_tag} baseline val pick_only={baseline} diag={baseline_diag}",
        flush=True,
    )
    print(
        f"[obs-full] starting train loop: {args.epochs} epochs, "
        f"{len(train_loader)} steps/ep (ep1/30 bar appears next)",
        flush=True,
    )

    history: list[dict] = []
    best_score = -1.0
    best_metrics = None
    start_time = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run = {
            "loss": 0.0,
            "loss_p": 0.0,
            "loss_s": 0.0,
            "noise_pen": 0.0,
            "wrong_p": 0.0,
            "wrong_s": 0.0,
            "ps_order": 0.0,
            "loss_nc": 0.0,
            "loss_exist": 0.0,
            "n": 0,
        }
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(
            train_loader,
            desc=f"ep{ep}/{args.epochs}",
            leave=True,
            file=_TQDM_FILE,
            ncols=100,
            mininterval=2.0,
            disable=False,
            dynamic_ncols=False,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                out = model(batch["x"], batch["t"])
                loss, stats = compute_loss(out, batch, args, model=model)
                loss = loss / max(1, args.grad_accum_steps)
            if not torch.isfinite(loss):
                print(f"[obs-full] skip non-finite loss ep{ep} step={step}", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            if not loss.requires_grad:
                print(f"[obs-full] skip no-grad loss ep{ep} step={step}", flush=True)
                continue
            scaler.scale(loss).backward()
            if step % max(1, args.grad_accum_steps) == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            bs = batch["x"].size(0)
            for k in (
                "loss",
                "loss_p",
                "loss_s",
                "noise_pen",
                "wrong_p",
                "wrong_s",
                "ps_order",
                "loss_nc",
                "loss_exist",
                "loss_p_res",
                "loss_rank",
            ):
                run[k] = run.get(k, 0.0) + stats.get(k, 0.0) * bs
            run["n"] += bs
            if run["n"] > 0:
                pbar.set_postfix(
                    loss=f"{run['loss'] / run['n']:.3f}",
                    p=f"{run['loss_p'] / run['n']:.2f}",
                    s=f"{run['loss_s'] / run['n']:.2f}",
                    refresh=False,
                )
            # Newline heartbeat so `tail -f` shows progress even when tqdm uses \r
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                print(
                    f"[obs-full] ep{ep}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={run['loss'] / max(1, run['n']):.3f}",
                    flush=True,
                )
        sched.step()
        train_stats = {k: run[k] / max(1, run["n"]) for k in run if k != "n"}
        row = {"epoch": ep, "train": train_stats, "lr": opt.param_groups[0]["lr"]}
        if ep % args.eval_every == 0 or ep == args.epochs:
            metrics = eval_pick_only(model, val_samples, device, args, obs_mod)
            diag = eval_pick_diagnostics(model, val_samples, device, args, obs_mod)
            exist_s = 0.0
            if isinstance(metrics.get("exist_acc"), dict):
                exist_s = float(metrics["exist_acc"].get("s", 0.0))
            if bool(getattr(args, "phase_exist", False)):
                score = 0.55 * metrics["p_f1"] + 0.30 * metrics["s_f1"] + 0.15 * exist_s
            else:
                score = 0.65 * metrics["p_f1"] + 0.35 * metrics["s_f1"]
            row["val_pick_only"] = metrics
            row["val_diag"] = diag
            row["val_score"] = score
            print(
                f"[obs-full] ep{ep} loss={train_stats['loss']:.4f} "
                f"val P={metrics['p_f1']:.3f} S={metrics['s_f1']:.3f} score={score:.3f} "
                f"existAccS={exist_s:.3f} diag={diag}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_metrics = metrics
                save_ckpt(
                    out_dir / "best.pt",
                    model,
                    args,
                    ep,
                    best_metrics,
                    history + [row],
                    extra={
                        "init": init_tag,
                        "baseline_val_pick_only": baseline,
                        "model_args": ckpt_args,
                        "n_train": train_samples_n,
                        "n_val": len(val_samples),
                        "n_holdout": len(holdout_samples),
                        "n_channels": int(args.input_dim),
                        "input_dim": int(args.input_dim),
                        "phase_exist": bool(getattr(args, "phase_exist", False)),
                    },
                )
        else:
            print(f"[obs-full] ep{ep} loss={train_stats['loss']:.4f}", flush=True)
        if ep % args.save_every == 0 or ep == args.epochs:
            save_ckpt(out_dir / "last.pt", model, args, ep, best_metrics, history + [row])
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_path = out_dir / "best.pt"
    if best_path.exists():
        best_model, _ = load_model(best_path, device, bypass_noise_cancel=False)
    else:
        best_model = model
    if args.split_json.strip() and not holdout_samples:
        print("[obs-full] loading deferred holdout for final eval...", flush=True)
        holdout_samples, hold_info, _ = load_split_samples(args.split_json.strip(), "holdout")
        holdout_samples = filter_alive_channels(
            holdout_samples, int(args.input_dim), mode="strict"
        )
        shrink_sample_waves(holdout_samples, int(args.input_dim))
        eval_pool = holdout_samples
        load_info["holdout_info"] = hold_info
    elif not eval_pool:
        eval_pool = val_samples
    final_metrics = eval_pick_only(best_model, eval_pool, device, args, obs_mod)
    final_diag = eval_pick_diagnostics(best_model, eval_pool, device, args, obs_mod)
    report = {
        "best_ckpt": str(best_path),
        "n_train": train_samples_n,
        "n_val": len(val_samples),
        "n_holdout": len(holdout_samples),
        "n_eval": len(eval_pool),
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "device": str(device),
        "load_info": load_info,
        "init": init_tag,
        "baseline_val_pick_only": baseline,
        "best_val_pick_only": best_metrics,
        "final_eval_pick_only": final_metrics,
        "final_eval_diag": final_diag,
        "elapsed_sec": time.time() - start_time,
        "args": vars(args),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "train_report.md").write_text(
        "\n".join(
            [
                "# OBS full retrain report",
                "",
                f"- best: `{best_path}`",
                f"- train/val/eval: {train_samples_n}/{len(val_samples)}/{len(eval_pool)}",
                f"- seq_len: {args.seq_len}",
                f"- best val pick-only: P={best_metrics['p_f1']:.3f} S={best_metrics['s_f1']:.3f}"
                if best_metrics
                else "- best val pick-only: N/A",
                f"- final eval pick-only: P={final_metrics['p_f1']:.3f} S={final_metrics['s_f1']:.3f}",
            ]
        )
        + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
