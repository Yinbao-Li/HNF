# -*- coding: utf-8 -*-
"""Training helpers for the foveated engine (two-stage strategy).

Stage 1 — behavior cloning:
  Supervise the Scheduler policy with expert gaze trajectories
  (e.g. from a dense run28 / short-window oracle).

Stage 2 — joint fine-tune:
  loss = pick_loss + λ_c * causal_consistency + λ_e * gaze_efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from hnf.foveated.engine import FoveatedEngine, FoveatedEngineOutput


@dataclass
class FoveatedTrainConfig:
    lambda_causal: float = 0.05
    lambda_efficiency: float = 0.02
    lambda_bc: float = 1.0
    max_gazes: int = 8


def pick_bce_loss(
    out: FoveatedEngineOutput,
    p_target: torch.Tensor,
    s_target: torch.Tensor,
) -> torch.Tensor:
    """p/s targets are (B, T) soft labels or one-hot peaks."""
    loss_p = F.binary_cross_entropy_with_logits(out.p_logits, p_target)
    loss_s = F.binary_cross_entropy_with_logits(out.s_logits, s_target)
    return 0.5 * (loss_p + loss_s)


def stage1_behavior_cloning_loss(
    engine: FoveatedEngine,
    features: torch.Tensor,
    expert_focus_idx: torch.Tensor,
    seq_len: int = 6000,
) -> torch.Tensor:
    """
    features: (B, 4) — e.g. [heat_peak, uncovered_mass, tip_unc, tip_time_norm]
    expert_focus_idx: (B,) integer sample indices
    """
    expert_norm = expert_focus_idx.float() / max(seq_len - 1, 1)
    return engine.scheduler.behavior_cloning_loss(features, expert_norm)


def stage2_joint_loss(
    engine: FoveatedEngine,
    out: FoveatedEngineOutput,
    p_target: torch.Tensor,
    s_target: torch.Tensor,
    cfg: Optional[FoveatedTrainConfig] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = cfg or FoveatedTrainConfig()
    l_pick = pick_bce_loss(out, p_target, s_target)
    l_causal = engine.causal_consistency_loss(out.edges)
    if not torch.is_tensor(l_causal):
        l_causal = torch.tensor(float(l_causal), device=out.p_logits.device)
    l_causal = l_causal.to(out.p_logits.device)
    l_eff = engine.gaze_efficiency_penalty(out.n_gazes, max_gazes=cfg.max_gazes)
    total = l_pick + cfg.lambda_causal * l_causal + cfg.lambda_efficiency * l_eff
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_pick": float(l_pick.detach().cpu()),
        "loss_causal": float(l_causal.detach().cpu()),
        "loss_efficiency": float(l_eff.detach().cpu()),
        "n_gazes_mean": float(out.n_gazes.float().mean().detach().cpu()),
    }
    return total, stats


def expert_focus_from_labels(
    p_idx: torch.Tensor,
    s_idx: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Simple oracle: alternate P then S peaks as expert gaze centers."""
    return torch.where(step % 2 == 0, p_idx, s_idx)
