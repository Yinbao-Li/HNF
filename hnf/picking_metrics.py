# -*- coding: utf-8 -*-
"""STEAD picking metrics aligned with EQTransformer / PhaseNet protocol."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PickingCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    mae_sec_sum: float = 0.0

    def prf(self) -> tuple[float, float, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return precision, recall, f1

    def mae_sec(self) -> float:
        return self.mae_sec_sum / max(self.tp, 1)


@dataclass
class DetectionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def prf(self) -> tuple[float, float, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return precision, recall, f1


@dataclass
class EvalAccumulator:
    det: DetectionCounts = field(default_factory=DetectionCounts)
    p: PickingCounts = field(default_factory=PickingCounts)
    s: PickingCounts = field(default_factory=PickingCounts)


def tolerance_bins(seq_len: int, pick_tolerance_sec: float) -> int:
    return max(1, round(pick_tolerance_sec * seq_len / 60.0))


def idx_to_sec(idx: int, seq_len: int) -> float:
    return float(idx) * 60.0 / float(seq_len)


def det_pred_from_logits(det_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Scalar or per-time detection logits -> per-trace bool."""
    det_prob = torch.sigmoid(det_logits)
    if det_prob.dim() == 1:
        return det_prob >= threshold
    return det_prob.amax(dim=-1) >= threshold


def apply_p_before_s_constraint(
    p_probs: torch.Tensor,
    s_probs: torch.Tensor,
    pick_threshold: float,
    min_gap_bins: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero sub-threshold peaks and enforce S after P when both exist."""
    p_out = p_probs.clone()
    s_out = s_probs.clone()
    for i in range(p_probs.size(0)):
        p_peak = p_probs[i].max().item()
        s_peak = s_probs[i].max().item()
        if p_peak < pick_threshold and s_peak < pick_threshold:
            continue
        p_idx = int(p_probs[i].argmax().item())
        s_idx = int(s_probs[i].argmax().item())
        if p_peak >= pick_threshold and s_peak >= pick_threshold and s_idx <= p_idx + min_gap_bins:
            s_out[i] = s_out[i] * 0.0
    return p_out, s_out


def update_detection_counts(
    acc: EvalAccumulator,
    det_pred: torch.Tensor,
    det_true: torch.Tensor,
) -> None:
    acc.det.tp += ((det_pred == 1) & (det_true == 1)).sum().item()
    acc.det.fp += ((det_pred == 1) & (det_true == 0)).sum().item()
    acc.det.fn += ((det_pred == 0) & (det_true == 1)).sum().item()


def update_picking_counts(
    counts: PickingCounts,
    probs: torch.Tensor,
    det_pred: torch.Tensor,
    det_true: torch.Tensor,
    valid: torch.Tensor,
    gt_idx: torch.Tensor,
    pick_threshold: float,
    tol_bins: int,
    seq_len: int,
) -> None:
    """
    EQTransformer-style picking:
      - peak must exceed threshold (no forced argmax)
      - event traces: evaluate only when det_pred is True
      - undetected events with GT count as FN
      - noise traces: peak above threshold counts as FP
    """
    max_prob, pred_idx = probs.max(dim=-1)

    for i in range(probs.size(0)):
        is_event = bool(det_true[i].item())
        is_noise = not is_event
        has_gt = bool(valid[i].item())
        detected = bool(det_pred[i].item())
        max_prob_i = max_prob[i].item()
        pred_exists = max_prob_i >= pick_threshold
        pred_idx_i = int(pred_idx[i].item())

        if is_noise:
            if pred_exists:
                counts.fp += 1
            continue

        if not has_gt:
            continue

        gt_i = int(gt_idx[i].item())
        within = pred_exists and abs(pred_idx_i - gt_i) <= tol_bins

        if not detected:
            counts.fn += 1
            continue

        if within:
            counts.tp += 1
            counts.mae_sec_sum += abs(idx_to_sec(pred_idx_i, seq_len) - idx_to_sec(gt_i, seq_len))
        elif pred_exists:
            counts.fn += 1
        else:
            counts.fn += 1


def finalize_metrics(acc: EvalAccumulator) -> dict[str, float]:
    det_pr, det_re, det_f1 = acc.det.prf()
    p_pr, p_re, p_f1 = acc.p.prf()
    s_pr, s_re, s_f1 = acc.s.prf()
    return {
        "det_precision": det_pr,
        "det_recall": det_re,
        "det_f1": det_f1,
        "p_precision": p_pr,
        "p_recall": p_re,
        "p_f1": p_f1,
        "p_mae_sec": acc.p.mae_sec(),
        "s_precision": s_pr,
        "s_recall": s_re,
        "s_f1": s_f1,
        "s_mae_sec": acc.s.mae_sec(),
    }


def picking_score(
    metrics: dict[str, float],
    mode: str = "mean",
    det_floor: float = 0.985,
) -> float:
    det = metrics["det_f1"]
    p = metrics["p_f1"]
    s = metrics["s_f1"]
    if mode == "mean":
        return (det + p + s) / 3.0
    if mode == "det_guard":
        pick_mean = 0.5 * (p + s)
        shortfall = max(0.0, det_floor - det)
        return pick_mean + 0.15 * det - 8.0 * shortfall
    if mode == "pick_focus":
        return 0.5 * (p + s)
    raise ValueError(f"unknown picking score mode: {mode}")
