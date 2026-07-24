#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig5 / Step-4 picking cross-dataset: OBS (SeisBench) zero-shot compare.

Models:
  - HNF STEAD-trained (run28 primary; run20 legacy optional)
  - EQTransformer / PhaseNet pretrained on STEAD  (fair zero-shot)
  - EQTransformer / PhaseNet pretrained on OBS   (domain-matched reference)

Multi-chunk OBS (201805–201808 by default). Protocol: pick-only F1, 0.5 s tol,
sigmoid on HNF logits, PhaseNet PSN, drop incomplete 3C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tools.analyze_stead_picking import load_model
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    tolerance_bins,
    update_detection_counts,
    update_picking_counts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS picking cross-dataset compare")
    p.add_argument(
        "--checkpoint",
        default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt",
    )
    p.add_argument("--hnf-label", default="", help="Result key for HNF (default inferred)")
    p.add_argument("--output-dir", default="outputs/obs_step4_run28_multichunk")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--chunks",
        default="201805,201806,201807,201808",
        help="Comma-separated OBS chunk ids (multi-chunk Step 4)",
    )
    p.add_argument(
        "--chunk",
        default="",
        help="Deprecated single-chunk alias; overrides --chunks if set",
    )
    p.add_argument("--max-events", type=int, default=800)
    p.add_argument("--seq-len", type=int, default=800, help="HNF resampled length over 60 s")
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument(
        "--p-offset-sec",
        type=float,
        default=8.0,
        help="P arrival target offset in the 60s window. STEAD-trained HNF sees P near ~6–8s; "
        "15s (old default) collapses P zero-shot via absolute-time prior mismatch.",
    )
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument(
        "--threshold-sweep",
        default="",
        help="Comma pick thresholds for HNF-only sweep after main eval (e.g. 0.15,0.2,0.3,0.4)",
    )
    p.add_argument("--skip-obs-pretrained", action="store_true",
                   help="Only fair zero-shot (HNF + EQT/PN STEAD)")
    p.add_argument("--skip-stead-pretrained", action="store_true",
                   help="Skip EQT(STEAD)/PhaseNet(STEAD) zero-shot baselines")
    p.add_argument(
        "--eqt-adapt-checkpoint",
        default="",
        help="Matched OBS light-adapt EQT ckpt (tools/train_obs_sb_light_adapt.py)",
    )
    p.add_argument(
        "--phasenet-adapt-checkpoint",
        default="",
        help="Matched OBS light-adapt PhaseNet ckpt",
    )
    p.add_argument("--eqt-adapt-label", default="EQT(STEAD+OBS-adapt)")
    p.add_argument("--phasenet-adapt-label", default="PhaseNet(STEAD+OBS-adapt)")
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--require-full-3c", action="store_true", default=True,
                   help="Keep only traces with energetic Z/1/2 (drop ZH/Z1H incomplete)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument(
        "--split-json",
        default="",
        help="If set, evaluate ONLY on disjoint holdout keys from tools/obs_matched_split.py",
    )
    return p.parse_args()


def _channel_alive(wave: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(wave ** 2, axis=-1))
    return rms > eps


def normalize_wave(wave: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel demean + scale. mode in {peak, std}."""
    x = wave.astype(np.float32, copy=True)
    x = x - x.mean(axis=-1, keepdims=True)
    if mode == "peak":
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
    elif mode == "std":
        scale = x.std(axis=-1, keepdims=True)
    else:
        raise ValueError(mode)
    return x / (scale + 1e-8)


def load_obs_windows(
    chunks: list[str],
    max_events: int,
    window_sec: float,
    p_offset_sec: float,
    seed: int,
    require_full_3c: bool = True,
    p_offset_min: float | None = None,
    p_offset_max: float | None = None,
):
    """Load OBS event windows.

    If p_offset_min < p_offset_max, each event gets a deterministic random
    offset in [min, max] (seeded) to kill the absolute-time prior shortcut.
    Otherwise all events use fixed ``p_offset_sec``.
    """
    import seisbench.data as sbd

    off_min = float(p_offset_sec if p_offset_min is None else p_offset_min)
    off_max = float(p_offset_sec if p_offset_max is None else p_offset_max)
    if off_max < off_min:
        off_min, off_max = off_max, off_min
    randomize_offset = (off_max - off_min) > 1e-8

    # Keep hydrophone so OBS-pretrained 4-channel models can run; land models use first 3.
    candidates: list[tuple[str, int]] = []
    per_chunk_cand: dict[str, int] = {}
    for chunk in chunks:
        ds = sbd.OBS(chunks=[chunk], download_if_missing=False, component_order="Z12H")
        meta = ds.metadata
        n_c = 0
        for i in range(len(ds)):
            row = meta.iloc[i]
            p = row.get("trace_p_arrival_sample")
            if p is None or (isinstance(p, float) and not np.isfinite(p)):
                continue
            # Prefer complete horizontal components for fair 3C land-model transfer.
            order = str(row.get("trace_component_order", ""))
            if require_full_3c and ("2" not in order):
                continue
            candidates.append((chunk, i))
            n_c += 1
        per_chunk_cand[chunk] = n_c

    rng = np.random.default_rng(seed)
    if len(candidates) > max_events:
        pick = rng.choice(len(candidates), size=max_events, replace=False)
        candidates = [candidates[int(i)] for i in sorted(pick.tolist())]

    # Cache dataset handles per chunk while materializing windows.
    ds_by_chunk = {
        c: sbd.OBS(chunks=[c], download_if_missing=False, component_order="Z12H")
        for c in chunks
    }

    samples = []
    n_drop_energy = 0
    per_chunk_kept: dict[str, int] = {c: 0 for c in chunks}
    for j, (chunk, i) in enumerate(candidates):
        ds = ds_by_chunk[chunk]
        wave, row = ds.get_sample(i)  # (4, npts) Z12H layout
        wave = np.asarray(wave, dtype=np.float32)
        sr = float(row.get("trace_sampling_rate_hz", 100.0))
        npts = wave.shape[-1]
        p_abs = float(row["trace_p_arrival_sample"])
        s_raw = row.get("trace_s_arrival_sample")
        s_abs = float(s_raw) if s_raw is not None and np.isfinite(float(s_raw)) else float("nan")

        if randomize_offset:
            # Independent of candidate list order: hash by (chunk, i).
            local = np.random.default_rng(seed + (hash((chunk, int(i))) % 1_000_000_007))
            this_offset = float(local.uniform(off_min, off_max))
        else:
            this_offset = float(off_min)

        win = int(round(window_sec * sr))
        start = int(round(p_abs - this_offset * sr))
        start = max(0, min(start, max(0, npts - win)))
        end = start + win
        if end > npts:
            pad = end - npts
            seg = np.pad(wave[:, start:npts], ((0, 0), (0, pad)), mode="constant")
        else:
            seg = wave[:, start:end]

        p_rel = p_abs - start
        s_rel = s_abs - start if np.isfinite(s_abs) else float("nan")
        p_valid = 0.0 <= p_rel < win
        s_valid = np.isfinite(s_rel) and 0.0 <= s_rel < win
        if not p_valid:
            continue

        alive = _channel_alive(seg)
        if require_full_3c and not bool(alive[:3].all()):
            n_drop_energy += 1
            continue

        event_key = f"{chunk}|{int(i)}"
        # Store RAW window; apply model-specific normalization at eval time.
        samples.append({
            "wave_4_raw": seg.copy(),
            "wave_3_raw": seg[:3].copy(),
            "sr": sr,
            "p_idx_native": int(round(p_rel)),
            "s_idx_native": int(round(s_rel)) if s_valid else -1,
            "p_valid": True,
            "s_valid": bool(s_valid),
            "p_offset_sec": this_offset,
            "ds_index": int(i),
            "event_key": event_key,
            "trace_name": str(row.get("trace_name_original", row.get("trace_name", i))),
            "station": str(row.get("station_code", "")),
            "component_order": str(row.get("trace_component_order", "")),
            "split": str(row.get("split", "")),
            "chunk": chunk,
        })
        per_chunk_kept[chunk] = per_chunk_kept.get(chunk, 0) + 1
    return samples, {
        "n_drop_energy": n_drop_energy,
        "n_candidate_idxs": len(candidates),
        "chunks": chunks,
        "per_chunk_candidates": per_chunk_cand,
        "per_chunk_kept": per_chunk_kept,
        "p_offset_min": off_min,
        "p_offset_max": off_max,
        "randomize_offset": randomize_offset,
    }


def load_obs_windows_from_entries(
    entries: list[dict],
    window_sec: float,
    require_full_3c: bool = True,
):
    """Rematerialize windows for a fixed list of {chunk, ds_index, p_offset_sec}."""
    import seisbench.data as sbd

    chunks = sorted({str(e["chunk"]) for e in entries})
    ds_by_chunk = {
        c: sbd.OBS(chunks=[c], download_if_missing=False, component_order="Z12H")
        for c in chunks
    }
    samples = []
    n_drop_energy = 0
    for e in entries:
        chunk = str(e["chunk"])
        i = int(e["ds_index"])
        this_offset = float(e["p_offset_sec"])
        ds = ds_by_chunk[chunk]
        wave, row = ds.get_sample(i)
        wave = np.asarray(wave, dtype=np.float32)
        sr = float(row.get("trace_sampling_rate_hz", 100.0))
        npts = wave.shape[-1]
        p_abs = float(row["trace_p_arrival_sample"])
        s_raw = row.get("trace_s_arrival_sample")
        s_abs = float(s_raw) if s_raw is not None and np.isfinite(float(s_raw)) else float("nan")
        win = int(round(window_sec * sr))
        start = int(round(p_abs - this_offset * sr))
        start = max(0, min(start, max(0, npts - win)))
        end = start + win
        if end > npts:
            pad = end - npts
            seg = np.pad(wave[:, start:npts], ((0, 0), (0, pad)), mode="constant")
        else:
            seg = wave[:, start:end]
        p_rel = p_abs - start
        s_rel = s_abs - start if np.isfinite(s_abs) else float("nan")
        p_valid = 0.0 <= p_rel < win
        s_valid = np.isfinite(s_rel) and 0.0 <= s_rel < win
        if not p_valid:
            continue
        alive = _channel_alive(seg)
        if require_full_3c and not bool(alive[:3].all()):
            n_drop_energy += 1
            continue
        event_key = f"{chunk}|{i}"
        samples.append({
            "wave_4_raw": seg.copy(),
            "wave_3_raw": seg[:3].copy(),
            "sr": sr,
            "p_idx_native": int(round(p_rel)),
            "s_idx_native": int(round(s_rel)) if s_valid else -1,
            "p_valid": True,
            "s_valid": bool(s_valid),
            "p_offset_sec": this_offset,
            "ds_index": i,
            "event_key": event_key,
            "trace_name": str(row.get("trace_name_original", row.get("trace_name", i))),
            "station": str(row.get("station_code", "")),
            "component_order": str(row.get("trace_component_order", "")),
            "split": str(row.get("split", "")),
            "chunk": chunk,
        })
    return samples, {"n_drop_energy": n_drop_energy, "n_entries": len(entries), "n_kept": len(samples)}



def to_hnf_batch(
    samples: list[dict],
    seq_len: int,
    window_sec: float,
    device: torch.device,
    n_channels: int = 3,
):
    xs, ts, p_idx, s_idx, p_valid, s_valid = [], [], [], [], [], []
    n_channels = int(n_channels)
    for s in samples:
        # Match STEADPickingDataset: per-channel demean + std, then resample.
        raw = s.get("wave_4_raw") if n_channels >= 4 else None
        if raw is None:
            raw = s["wave_3_raw"]
        wave = np.asarray(raw[:n_channels], dtype=np.float32)
        x = torch.from_numpy(normalize_wave(wave, "std")).float()  # (C, Tn)
        x = F.interpolate(x.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)  # (T,C)
        scale = seq_len / float(wave.shape[-1])
        xs.append(x)
        ts.append(torch.linspace(0.0, window_sec, seq_len).unsqueeze(-1))
        p_idx.append(int(round(s["p_idx_native"] * scale)))
        s_idx.append(int(round(s["s_idx_native"] * scale)) if s["s_valid"] else 0)
        p_valid.append(1.0)
        s_valid.append(1.0 if s["s_valid"] else 0.0)
    return (
        torch.stack(xs).to(device),
        torch.stack(ts).to(device),
        torch.tensor(p_idx, device=device),
        torch.tensor(s_idx, device=device),
        torch.tensor(p_valid, device=device),
        torch.tensor(s_valid, device=device),
    )


def to_sb_batch(samples: list[dict], device: torch.device, n_channels: int, norm_mode: str):
    xs = []
    for s in samples:
        raw = s.get("wave_4_raw") if n_channels >= 4 else None
        if raw is None:
            raw = s["wave_3_raw"]
        xs.append(torch.from_numpy(normalize_wave(np.asarray(raw[:n_channels], dtype=np.float32), norm_mode)).float())
    tlen = max(x.shape[-1] for x in xs)
    out = []
    for x in xs:
        if x.shape[-1] < tlen:
            x = F.pad(x, (0, tlen - x.shape[-1]))
        out.append(x)
    x = torch.stack(out).to(device)
    p_idx = torch.tensor([s["p_idx_native"] for s in samples], device=device)
    s_idx = torch.tensor([s["s_idx_native"] if s["s_valid"] else 0 for s in samples], device=device)
    p_valid = torch.tensor([1.0] * len(samples), device=device)
    s_valid = torch.tensor([1.0 if s["s_valid"] else 0.0 for s in samples], device=device)
    return x, p_idx, s_idx, p_valid, s_valid


def _pick_only_counts(
    probs,
    valid,
    gt_idx,
    pick_th,
    tol_bins,
    seq_len,
    counts,
    exist_prob=None,
    exist_th: float = 0.5,
    score_absent: bool = False,
    gate_mode: str = "hard",
    soft_th: float = 0.25,
    decode_mode: str = "argmax",
    decode_compete_ratio: float = 0.70,
    decode_late_penalty: float = 0.0,
    pred_idx=None,
    peak_score=None,
    decode_echo_gap_lo_bins: int = 20,
    decode_echo_gap_hi_bins: int = 300,
    decode_echo_ratio: float = 0.70,
    decode_echo_penalty: float = 0.35,
    decode_onset_bonus: float = 0.10,
):
    """Event-window pick-only: peak>th (and optional exist gate) within tol = TP.

    gate_mode:
      - hard: peak>=pick_th AND exist>=exist_th
      - soft: (peak * exist) >= soft_th
      - soft_floor: soft AND exist>=exist_th

    If score_absent=True, also score windows with valid=False:
      - pred exists → FP
      - pred absent → TN (ignored in F1; tracked on counts.tn if present)

    decode_mode: see hnf.pick_decode.decode_pick_index (default argmax).
    pred_idx / peak_score: optional precomputed (B,) overrides (e.g. after residual).
    """
    from hnf.picking_metrics import idx_to_sec
    from hnf.pick_decode import decode_pick_index

    for i in range(probs.size(0)):
        if pred_idx is not None:
            pred_i = int(pred_idx[i].item() if hasattr(pred_idx[i], "item") else pred_idx[i])
            if peak_score is not None:
                pk = float(peak_score[i].item() if hasattr(peak_score[i], "item") else peak_score[i])
            else:
                pk = float(probs[i, max(0, min(pred_i, probs.size(-1) - 1))].item())
        else:
            pred_i, pk = decode_pick_index(
                probs[i],
                pick_th=pick_th,
                mode=decode_mode,
                compete_ratio=decode_compete_ratio,
                late_penalty=decode_late_penalty,
                echo_gap_lo_bins=decode_echo_gap_lo_bins,
                echo_gap_hi_bins=decode_echo_gap_hi_bins,
                echo_ratio=decode_echo_ratio,
                echo_penalty=decode_echo_penalty,
                onset_bonus=decode_onset_bonus,
            )
        if exist_prob is None:
            pred_exists = pk >= pick_th
        else:
            ex = float(exist_prob[i])
            if gate_mode == "soft":
                pred_exists = (pk * ex) >= soft_th
            elif gate_mode == "soft_floor":
                pred_exists = ((pk * ex) >= soft_th) and (ex >= exist_th)
            else:
                pred_exists = (pk >= pick_th) and (ex >= exist_th)
        has_gt = bool(valid[i].item())
        if not has_gt:
            if score_absent:
                if pred_exists:
                    counts.fp += 1
                elif hasattr(counts, "tn"):
                    counts.tn += 1
            continue
        gt_i = int(gt_idx[i])
        if pred_exists and abs(pred_i - gt_i) <= tol_bins:
            counts.tp += 1
            counts.mae_sec_sum += abs(idx_to_sec(pred_i, seq_len) - idx_to_sec(gt_i, seq_len))
        elif pred_exists:
            counts.fp += 1
            counts.fn += 1
        else:
            counts.fn += 1


@torch.no_grad()
def eval_hnf(
    model,
    samples,
    device,
    seq_len,
    window_sec,
    pick_th,
    det_th,
    tol_sec,
    batch_size,
    n_channels: int = 3,
    exist_th: float = 0.5,
    score_absent: bool = False,
    gate_mode: str = "hard",
    soft_th: float = 0.25,
    p_decode_mode: str = "argmax",
    s_decode_mode: str = "argmax",
    decode_compete_ratio: float = 0.70,
    decode_late_penalty: float = 0.0,
    apply_p_residual: bool = False,
    apply_causal_peak_rank: bool = False,
    decode_echo_gap_lo_bins: int = 20,
    decode_echo_gap_hi_bins: int = 300,
    decode_echo_ratio: float = 0.70,
    decode_echo_penalty: float = 0.35,
    decode_onset_bonus: float = 0.10,
):
    """HNF forward returns LOGITS in det/p/s — must sigmoid before threshold (STEAD protocol)."""
    from hnf.pick_decode import decode_pick_indices_batch

    acc = EvalAccumulator()
    pick_acc = EvalAccumulator()
    tol = tolerance_bins(seq_len, tol_sec)
    n_det_nan = 0
    n_pick_nan = 0
    n_channels = int(getattr(model, "input_dim", n_channels) or n_channels)
    exist_correct = {"p": 0, "s": 0}
    exist_total = {"p": 0, "s": 0}
    use_p_res = bool(apply_p_residual) and getattr(model, "p_residual_offset", None) is not None
    use_causal_rank = bool(apply_causal_peak_rank) and getattr(model, "p_causal_peak_rank", None) is not None
    decode_kw = dict(
        compete_ratio=decode_compete_ratio,
        late_penalty=decode_late_penalty,
        echo_gap_lo_bins=decode_echo_gap_lo_bins,
        echo_gap_hi_bins=decode_echo_gap_hi_bins,
        echo_ratio=decode_echo_ratio,
        echo_penalty=decode_echo_penalty,
        onset_bonus=decode_onset_bonus,
    )
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x, t, p_idx, s_idx, p_valid, s_valid = to_hnf_batch(
            chunk, seq_len, window_sec, device, n_channels=n_channels
        )
        out = model(x, t)
        det_true = torch.ones(len(chunk), device=device)
        if "det_logits" in out:
            det_logits = out["det_logits"]
            p_logits = out["p_logits"]
            s_logits = out["s_logits"]
        else:
            # run20 / picking_model: keys are det/p/s but values are logits
            det_logits = out["det"]
            p_logits = out["p"]
            s_logits = out["s"]

        pick_bad = (~torch.isfinite(p_logits)).any(dim=-1) | (~torch.isfinite(s_logits)).any(dim=-1)
        n_pick_nan += int(pick_bad.sum().item())
        p_logits = torch.nan_to_num(p_logits, nan=-50.0, posinf=50.0, neginf=-50.0)
        s_logits = torch.nan_to_num(s_logits, nan=-50.0, posinf=50.0, neginf=-50.0)
        p_probs = torch.sigmoid(p_logits)
        s_probs = torch.sigmoid(s_logits)

        p_exist = None
        s_exist = None
        if "p_exist" in out and "s_exist" in out:
            p_exist = torch.sigmoid(torch.nan_to_num(out["p_exist"], nan=-50.0))
            s_exist = torch.sigmoid(torch.nan_to_num(out["s_exist"], nan=-50.0))
            for i in range(len(chunk)):
                exist_total["p"] += 1
                exist_total["s"] += 1
                if bool((p_exist[i] >= exist_th) == (p_valid[i] > 0.5)):
                    exist_correct["p"] += 1
                if bool((s_exist[i] >= exist_th) == (s_valid[i] > 0.5)):
                    exist_correct["s"] += 1

        if det_logits.dim() == 1:
            det_nan = ~torch.isfinite(det_logits)
            n_det_nan += int(det_nan.sum().item())
            det_prob = torch.sigmoid(torch.nan_to_num(det_logits, nan=-50.0))
            det_pred = det_prob >= det_th
        else:
            det_nan = (~torch.isfinite(det_logits)).any(dim=-1)
            n_det_nan += int(det_nan.sum().item())
            det_prob = torch.sigmoid(torch.nan_to_num(det_logits, nan=-50.0))
            det_pred = det_prob.amax(dim=-1) >= det_th

        p_probs, s_probs = apply_p_before_s_constraint(p_probs, s_probs, pick_th)
        update_detection_counts(acc, det_pred, det_true)
        update_picking_counts(acc.p, p_probs, det_pred, det_true, p_valid, p_idx, pick_th, tol, seq_len)
        update_picking_counts(acc.s, s_probs, det_pred, det_true, s_valid, s_idx, pick_th, tol, seq_len)
        # pick-only: ignore detection gate (OBS often NaNs det head)
        force_det = torch.ones_like(det_pred)

        p_pred_idx = None
        p_peak_sc = None
        p_field = out.get("p_field_env")
        if use_causal_rank:
            rho = out.get("rho")
            if rho is None:
                rho = torch.zeros_like(p_probs)
            p_pred_idx, p_peak_sc = model.p_causal_peak_rank.decode(
                p_probs,
                p_field if p_field is not None else p_probs,
                rho,
                pick_th=pick_th,
                compete_ratio=0.0,
            )
        elif use_p_res or str(p_decode_mode) != "argmax":
            p_pred_idx, p_peak_sc = decode_pick_indices_batch(
                p_probs,
                pick_th=pick_th,
                mode=p_decode_mode,
                field_env=p_field,
                **decode_kw,
            )
            if use_p_res:
                p_pred_idx = model.refine_p_indices(x, t, p_pred_idx)
                # Keep gate score from coarse decode peak (existence), not shifted bin.
        _pick_only_counts(
            p_probs, p_valid, p_idx, pick_th, tol, seq_len, pick_acc.p,
            exist_prob=p_exist, exist_th=exist_th, score_absent=score_absent,
            gate_mode=gate_mode, soft_th=soft_th,
            decode_mode=p_decode_mode,
            decode_compete_ratio=decode_compete_ratio,
            decode_late_penalty=decode_late_penalty,
            pred_idx=p_pred_idx,
            peak_score=p_peak_sc,
            decode_echo_gap_lo_bins=decode_echo_gap_lo_bins,
            decode_echo_gap_hi_bins=decode_echo_gap_hi_bins,
            decode_echo_ratio=decode_echo_ratio,
            decode_echo_penalty=decode_echo_penalty,
            decode_onset_bonus=decode_onset_bonus,
        )
        _pick_only_counts(
            s_probs, s_valid, s_idx, pick_th, tol, seq_len, pick_acc.s,
            exist_prob=s_exist, exist_th=exist_th, score_absent=score_absent,
            gate_mode=gate_mode, soft_th=soft_th,
            decode_mode=s_decode_mode,
            decode_compete_ratio=decode_compete_ratio,
            decode_late_penalty=decode_late_penalty,
            decode_echo_gap_lo_bins=decode_echo_gap_lo_bins,
            decode_echo_gap_hi_bins=decode_echo_gap_hi_bins,
            decode_echo_ratio=decode_echo_ratio,
            decode_echo_penalty=decode_echo_penalty,
            decode_onset_bonus=decode_onset_bonus,
        )
        update_detection_counts(pick_acc, force_det, det_true)
    coupled = finalize_metrics(acc)
    pick_only = finalize_metrics(pick_acc)
    exist_acc = {
        "p": exist_correct["p"] / max(exist_total["p"], 1),
        "s": exist_correct["s"] / max(exist_total["s"], 1),
        "n": exist_total["p"],
    }
    return {
        "coupled": coupled,
        "pick_only": pick_only,
        "exist_acc": exist_acc,
        "exist_th": exist_th,
        "gate_mode": gate_mode,
        "soft_th": soft_th,
        "n_det_nan": n_det_nan,
        "n_pick_nan": n_pick_nan,
        "note": "sigmoid on logits; optional p_exist/s_exist gate when heads present",
    }


@torch.no_grad()
def eval_seisbench(
    model,
    samples,
    device,
    pick_th,
    det_th,
    tol_sec,
    batch_size,
    kind: str,
    n_channels: int,
    norm_mode: str,
):
    acc = EvalAccumulator()
    pick_acc = EvalAccumulator()
    native_len = (
        samples[0].get("wave_4_raw", samples[0].get("wave_3_raw"))
    ).shape[-1]
    tol = max(1, int(round(tol_sec * samples[0]["sr"])))
    model = model.to(device).eval()
    # PhaseNet labels are "PSN" in SeisBench (P, S, Noise) — NOT NPS.
    labels = "".join(getattr(model, "labels", "PSN") or "PSN")
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        x, p_idx, s_idx, p_valid, s_valid = to_sb_batch(
            chunk, device, n_channels=n_channels, norm_mode=norm_mode
        )
        out = model(x)
        if kind == "eqt":
            det_prob, p_prob, s_prob = out
            if det_prob.dim() == 3:
                det_prob = det_prob.squeeze(1)
            if p_prob.dim() == 3:
                p_prob = p_prob.squeeze(1)
            if s_prob.dim() == 3:
                s_prob = s_prob.squeeze(1)
            det_pred = det_prob.amax(dim=-1) >= det_th
        else:
            if out.dim() != 3 or out.shape[1] != 3:
                raise RuntimeError(f"Unexpected PhaseNet out shape {tuple(out.shape)}")
            if labels.upper().startswith("PS"):
                p_prob, s_prob = out[:, 0], out[:, 1]
            elif labels.upper().startswith("NP"):
                p_prob, s_prob = out[:, 1], out[:, 2]
            else:
                # fallback: assume PSN
                p_prob, s_prob = out[:, 0], out[:, 1]
            det_pred = torch.maximum(p_prob.amax(-1), s_prob.amax(-1)) >= pick_th
        det_true = torch.ones(len(chunk), device=device)
        p_prob, s_prob = apply_p_before_s_constraint(p_prob, s_prob, pick_th)
        update_detection_counts(acc, det_pred, det_true)
        update_picking_counts(acc.p, p_prob, det_pred, det_true, p_valid, p_idx, pick_th, tol, native_len)
        update_picking_counts(acc.s, s_prob, det_pred, det_true, s_valid, s_idx, pick_th, tol, native_len)
        force_det = torch.ones_like(det_pred)
        _pick_only_counts(p_prob, p_valid, p_idx, pick_th, tol, native_len, pick_acc.p)
        _pick_only_counts(s_prob, s_valid, s_idx, pick_th, tol, native_len, pick_acc.s)
        update_detection_counts(pick_acc, force_det, det_true)
    return {
        "coupled": finalize_metrics(acc),
        "pick_only": finalize_metrics(pick_acc),
        "norm_mode": norm_mode,
        "labels": labels,
    }


def load_sb_model(name: str, weights: str):
    import seisbench.models as sbm

    cls = getattr(sbm, name)
    return cls.from_pretrained(weights)


def plot_compare(results: dict, out_dir: Path) -> str:
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    p_f1 = [results[n]["pick_only"]["p_f1"] for n in names]
    s_f1 = [results[n]["pick_only"]["s_f1"] for n in names]
    p_mae = [results[n]["pick_only"]["p_mae_sec"] for n in names]
    s_mae = [results[n]["pick_only"]["s_mae_sec"] for n in names]
    colors = ["C2" if "HNF" in n else ("C3" if "(OBS)" in n else "C0") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w / 2, p_f1, width=w, label="P-F1", color="C0")
    axes[0].bar(x + w / 2, s_f1, width=w, label="S-F1", color="C1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("F1 (pick-only)")
    axes[0].set_title("OBS event-window picking F1")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x - w / 2, p_mae, width=w, label="P-MAE", color="C0")
    axes[1].bar(x + w / 2, s_mae, width=w, label="S-MAE", color="C1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    axes[1].set_ylabel("MAE (s)")
    axes[1].set_title("Timing error on true positives")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("OBS multi-chunk picking: HNF vs EQT vs PhaseNet (pick-only)", fontsize=12)
    p = out_dir / "obs_picking_compare.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    (docs / "fig5_obs_picking_compare.png").write_bytes(p.read_bytes())
    return str(p)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.chunk.strip():
        chunks = [args.chunk.strip()]
    else:
        chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    ckpt_path = Path(args.checkpoint)
    if args.hnf_label:
        hnf_key = args.hnf_label
    elif "run28" in str(ckpt_path):
        hnf_key = "HNF(run28/STEAD)"
    elif "run20" in str(ckpt_path):
        hnf_key = "HNF(run20/STEAD)"
    else:
        hnf_key = f"HNF({ckpt_path.parent.name})"

    print(f"[obs-compare] loading OBS chunks={chunks}", flush=True)
    if args.split_json.strip():
        from tools.obs_matched_split import load_split, load_split_samples

        split_meta = load_split(args.split_json.strip())
        if "holdout_entries" in split_meta:
            samples, load_info, _ = load_split_samples(args.split_json.strip(), "holdout")
            load_info = {
                **load_info,
                "split_json": args.split_json.strip(),
                "eval_mode": "disjoint_holdout",
                "protocol": split_meta.get("protocol"),
                "p_offset_min": split_meta.get("p_offset_min"),
                "p_offset_max": split_meta.get("p_offset_max"),
                "n_holdout": len(samples),
            }
        else:
            from tools.obs_matched_split import filter_by_keys

            pool_n = int(split_meta["train_n"]) + int(split_meta["holdout_n"])
            samples_all, load_info = load_obs_windows(
                chunks, pool_n, args.window_sec, args.p_offset_sec,
                int(split_meta["seed"]), require_full_3c=args.require_full_3c,
            )
            samples = filter_by_keys(samples_all, split_meta["holdout_keys"])
            load_info = {
                **load_info,
                "split_json": args.split_json.strip(),
                "eval_mode": "disjoint_holdout",
                "n_holdout": len(samples),
            }
        print(
            f"[obs-compare] DISJOINT holdout n={len(samples)} from {args.split_json} "
            f"protocol={split_meta.get('protocol','fixed')}",
            flush=True,
        )
    else:
        samples, load_info = load_obs_windows(
            chunks, args.max_events, args.window_sec, args.p_offset_sec, args.seed,
            require_full_3c=args.require_full_3c,
        )
    n_s = sum(1 for s in samples if s["s_valid"])
    print(
        f"[obs-compare] n={len(samples)} with_S={n_s} device={device} "
        f"drop_energy={load_info.get('n_drop_energy')} "
        f"candidates={load_info.get('n_candidate_idxs', load_info.get('n_entries'))}",
        flush=True,
    )
    if len(samples) < 10:
        raise RuntimeError("Too few OBS samples")

    results = {}

    print(f"[obs-compare] {hnf_key} ckpt={ckpt_path}...", flush=True)
    hnf, _ = load_model(ckpt_path, device, bypass_noise_cancel=False)
    # Match EQT-like long sequences: sparse band + capped light-cone if requested / stored.
    try:
        raw = torch.load(ckpt_path, map_location="cpu")
        aa = raw.get("adapt_args") or {}
        force_sparse = bool(aa.get("force_sparse_band")) or args.seq_len >= 3000
        cap_local = float(aa.get("cap_local_window_sec") or (3.0 if args.seq_len >= 3000 else 0.0))
        if force_sparse or cap_local > 0:
            n_sp = n_cap = 0
            for mod in hnf.modules():
                if force_sparse and hasattr(mod, "sparse_band"):
                    mod.sparse_band = True
                    n_sp += 1
                if cap_local > 0 and hasattr(mod, "local_window_sec") and mod.local_window_sec is not None:
                    before = float(mod.local_window_sec)
                    mod.local_window_sec = min(before, cap_local)
                    if mod.local_window_sec < before:
                        n_cap += 1
            print(
                f"[obs-compare] long-seq cfg sparse={force_sparse}({n_sp}) "
                f"cap_local={cap_local}(capped={n_cap}) seq_len={args.seq_len}",
                flush=True,
            )
    except Exception as e:
        print(f"[obs-compare] long-seq cfg skip: {e}", flush=True)
    results[hnf_key] = eval_hnf(
        hnf, samples, device, args.seq_len, args.window_sec,
        args.pick_threshold, args.det_threshold, args.tol_sec, args.batch_size,
    )
    print(results[hnf_key]["pick_only"], flush=True)

    # STEAD land models: SeisBench default norm=peak; OBS models: norm=std.
    sb_specs = []
    if not args.skip_stead_pretrained:
        sb_specs += [
            ("EQT(STEAD)", "EQTransformer", "stead", "eqt", 3, "peak"),
            ("PhaseNet(STEAD)", "PhaseNet", "stead", "phasenet", 3, "peak"),
        ]
    if not args.skip_obs_pretrained:
        sb_specs += [
            ("EQT(OBS)", "EQTransformer", "obs", "eqt", 4, "std"),
            ("PhaseNet(OBS)", "PhaseNet", "obs", "phasenet", 4, "std"),
        ]
    for label, name, weights, kind, nch, norm_mode in sb_specs:
        print(f"[obs-compare] {label} (norm={norm_mode})...", flush=True)
        try:
            m = load_sb_model(name, weights)
            # Prefer model.norm if present
            model_norm = getattr(m, "norm", None)
            if model_norm in ("peak", "std"):
                norm_mode = model_norm
            results[label] = eval_seisbench(
                m, samples, device, args.pick_threshold, args.det_threshold,
                args.tol_sec, args.batch_size, kind=kind, n_channels=nch,
                norm_mode=norm_mode,
            )
            print(results[label]["pick_only"], flush=True)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print("  failed:", e, flush=True)

    def _eval_sb_adapt(label: str, ckpt_path: str, name: str, kind: str) -> None:
        print(f"[obs-compare] {label} ckpt={ckpt_path}...", flush=True)
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            m = load_sb_model(name, ckpt.get("weights_init", "stead"))
            m.load_state_dict(ckpt["state_dict"])
            nch = int(ckpt.get("n_channels", 3))
            norm_mode = ckpt.get("norm_mode") or getattr(m, "norm", None) or "peak"
            results[label] = eval_seisbench(
                m, samples, device, args.pick_threshold, args.det_threshold,
                args.tol_sec, args.batch_size, kind=kind, n_channels=nch,
                norm_mode=norm_mode,
            )
            print(results[label]["pick_only"], flush=True)
        except Exception as e:
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            print("  failed:", e, flush=True)

    if args.eqt_adapt_checkpoint.strip():
        _eval_sb_adapt(
            args.eqt_adapt_label, args.eqt_adapt_checkpoint.strip(),
            "EQTransformer", "eqt",
        )
    if args.phasenet_adapt_checkpoint.strip():
        _eval_sb_adapt(
            args.phasenet_adapt_label, args.phasenet_adapt_checkpoint.strip(),
            "PhaseNet", "phasenet",
        )

    thr_sweep = {}
    if args.threshold_sweep.strip():
        thr_list = [float(x) for x in args.threshold_sweep.split(",") if x.strip()]
        print(f"[obs-compare] HNF threshold sweep {thr_list}...", flush=True)
        for thr in thr_list:
            thr_sweep[f"{thr:.3f}"] = eval_hnf(
                hnf, samples, device, args.seq_len, args.window_sec,
                thr, args.det_threshold, args.tol_sec, args.batch_size,
            )["pick_only"]
            print(f"  thr={thr:.3f}", thr_sweep[f"{thr:.3f}"], flush=True)

    ok = {k: v for k, v in results.items() if "error" not in v}
    fig = plot_compare(ok, out_dir)
    hnf_is_adapt = "adapt" in hnf_key.lower()
    zs_keys = [k for k in ("EQT(STEAD)", "PhaseNet(STEAD)") if k in results]
    if not hnf_is_adapt:
        zs_keys = [hnf_key] + zs_keys
    # Matched light-adapt cohort (same STEAD→OBS head-adapt budget).
    matched_adapt_keys = [
        k for k in (
            hnf_key if hnf_is_adapt else None,
            args.eqt_adapt_label if args.eqt_adapt_checkpoint.strip() else None,
            args.phasenet_adapt_label if args.phasenet_adapt_checkpoint.strip() else None,
        ) if k and k in results
    ]
    domain_keys = [k for k in ("EQT(OBS)", "PhaseNet(OBS)") if k in results]
    if hnf_is_adapt and hnf_key not in matched_adapt_keys:
        domain_keys = [hnf_key] + domain_keys

    def _row(k: str) -> str:
        v = results.get(k)
        if v is None:
            return ""
        if "error" in v:
            return f"| `{k}` | ERR | | | |"
        po = v["pick_only"]
        return (
            f"| `{k}` | {po['p_f1']:.3f} | {po['s_f1']:.3f} | "
            f"{po['p_mae_sec']:.3f} | {po['s_mae_sec']:.3f} |"
        )

    report = {
        "dataset": "SeisBench OBS",
        "eval_domain": "OBS",
        "chunks": chunks,
        "n_events": len(samples),
        "n_with_s": n_s,
        "load_info": load_info,
        "require_full_3c": args.require_full_3c,
        "tol_sec": args.tol_sec,
        "pick_threshold": args.pick_threshold,
        "det_threshold": args.det_threshold,
        "window_sec": args.window_sec,
        "hnf_checkpoint": str(ckpt_path),
        "hnf_treatment": "obs-adapt" if hnf_is_adapt else "zero-shot",
        "device": str(device),
        "results": results,
        "hnf_threshold_sweep": thr_sweep,
        "figure": fig,
        "protocol_fixes": [
            "PhaseNet labels are PSN (was wrongly decoded as NPS)",
            "STEAD models use peak-norm; OBS models use std-norm (SeisBench defaults)",
            "HNF uses per-channel demean+std then resample (matches STEAD training)",
            "Drop incomplete 3C traces (ZH/Z1H) for fair land-model transfer",
            "HNF forward returns logits; must sigmoid before threshold (matches train_stead_picking.py)",
            "Primary metric: pick-only F1 on event windows",
            "Step 4: multi-chunk sample pool before capped max-events",
            "Fairness: compare only within same treatment on OBS (all ZS or all OBS-exposed)",
        ],
        "notes": {
            "fair_zero_shot_obs_eval": zs_keys,
            "fair_matched_light_adapt": matched_adapt_keys,
            "obs_pretrained_reference": domain_keys,
            "do_not_cross_compare": (
                "Compare only within A (all ZS) or B (matched light-adapt). "
                "EQT(OBS)/PhaseNet(OBS) are full OBS-pretrained refs (4C), not same-budget adapt."
            ),
            "primary_metric": "pick_only on event windows (detection gate disabled)",
            "secondary_metric": "coupled EQT-style detection+picking",
            "channels": "land/HNF/STEAD models use Z12; OBS-pretrained use Z12H",
        },
    }
    (out_dir / "obs_picking_compare_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# OBS Picking Cross-Dataset Compare (Step 4)",
        "",
        f"- eval domain: **OBS** (all rows below)",
        f"- chunks: `{','.join(chunks)}`",
        f"- n: {len(samples)} (with S: {n_s})",
        f"- HNF ckpt: `{ckpt_path}`",
        f"- HNF treatment: `{'OBS-adapt' if hnf_is_adapt else 'zero-shot'}`",
        f"- device: `{device}`",
        f"- tolerance: {args.tol_sec} s",
        "- primary: **pick-only** F1 on event windows",
        "",
        "## Fairness rule",
        "- Compare only **same treatment** on OBS: all ZS, or all OBS-exposed.",
        "- Do **not** claim adapt-HNF vs STEAD→OBS ZS in one leaderboard.",
        "",
        "## A. Zero-shot (train=STEAD → eval=OBS)",
        "",
        "| Model | P-F1 | S-F1 | P-MAE | S-MAE |",
        "|------|-----:|-----:|------:|------:|",
    ]
    for k in zs_keys:
        row = _row(k)
        if row:
            md.append(row)
    if hnf_is_adapt:
        md.append("")
        md.append(f"_HNF treatment is OBS-adapt (`{hnf_key}`); excluded from table A — see B._")
    md += [
        "",
        "## B. Matched light-adapt (STEAD init → OBS heads; eval=OBS)",
        "",
        "| Model | P-F1 | S-F1 | P-MAE | S-MAE |",
        "|------|-----:|-----:|------:|------:|",
    ]
    for k in matched_adapt_keys:
        row = _row(k)
        if row:
            md.append(row)
    if not matched_adapt_keys:
        md.append("_No matched adapt checkpoints provided._")
    md += [
        "",
        "## C. Full OBS-pretrained reference (not same-budget)",
        "",
        "| Model | P-F1 | S-F1 | P-MAE | S-MAE | note |",
        "|------|-----:|-----:|------:|------:|------|",
    ]
    domain_notes = {
        "EQT(OBS)": "full OBS-pretrained (4C) reference",
        "PhaseNet(OBS)": "full OBS-pretrained (4C) reference",
    }
    for k in domain_keys:
        v = results.get(k)
        if v is None:
            continue
        if "error" in v:
            md.append(f"| `{k}` | ERR | | | | |")
        else:
            po = v["pick_only"]
            md.append(
                f"| `{k}` | {po['p_f1']:.3f} | {po['s_f1']:.3f} | "
                f"{po['p_mae_sec']:.3f} | {po['s_mae_sec']:.3f} | "
                f"{domain_notes.get(k, '')} |"
            )
    if thr_sweep:
        md += ["", "## HNF pick-threshold sweep", "", "| thr | P-F1 | S-F1 |", "|----:|-----:|-----:|"]
        for thr, po in thr_sweep.items():
            md.append(f"| {thr} | {po['p_f1']:.3f} | {po['s_f1']:.3f} |")
    md += [
        "",
        "## Protocol",
        "- PhaseNet channel order: **PSN**",
        "- Normalization: STEAD models `peak`, OBS models `std`, HNF per-channel `std`",
        "- Sample filter: require full energetic Z/1/2 (drop ZH/Z1H incompletes)",
        "- Multi-chunk pool then capped sampling for Step 4",
        "",
        "## Interpretation",
        "- Table A: fair land→OBS zero-shot.",
        "- Table B: matched light-adapt (same OBS split/budget; head-only).",
        "- Table C: full OBS-pretrained upper reference (different training budget/channels).",
        "",
        "## Figure",
        f"- `{Path(fig).name}`",
    ]
    (out_dir / "obs_picking_compare_report.md").write_text("\n".join(md))
    slim = {}
    for k, v in results.items():
        if "error" in v:
            slim[k] = v
        else:
            slim[k] = {kk: v["pick_only"][kk] for kk in ("p_f1", "s_f1", "p_mae_sec", "s_mae_sec")}
    print(json.dumps({
        "n": len(samples),
        "chunks": chunks,
        "results_pick_only": slim,
        "hnf_threshold_sweep": {
            k: {kk: v[kk] for kk in ("p_f1", "s_f1")} for k, v in thr_sweep.items()
        },
        "figure": fig,
    }, indent=2))


if __name__ == "__main__":
    main()
