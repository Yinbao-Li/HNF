# -*- coding: utf-8 -*-
"""Dual-path inversion: geo head for STEAD, macro baseline for synthetic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

from hnf.zhizi_inversion_bridge import ZhiziInversionBridge, load_inversion_bridge_from_checkpoint

DEFAULT_SYNTH_HEAD = "outputs/zhizi_inversion_bridge_macro/best_physics_head.pt"
DEFAULT_STEAD_HEAD = "outputs/zhizi_inversion_mixed_geo/best_physics_head.pt"


@dataclass
class DualPathInversionBridge:
    """
    Route STEAD events through geo-conditioned head; synthetic through macro baseline.

    Causal: real geometry only enters STEAD path; synth path has no geo conditioning.
    """

    stead: ZhiziInversionBridge
    synth: ZhiziInversionBridge
    stead_head: str
    synth_head: str

    @property
    def geo_condition(self) -> bool:
        return getattr(self.stead, "geo_condition", False)


def load_dual_path_bridge(
    backbone: nn.Module,
    device,
    *,
    stead_head: str = DEFAULT_STEAD_HEAD,
    synth_head: str = DEFAULT_SYNTH_HEAD,
    embed_dim: int = 64,
    n_layers: int = 5,
    infer_seq_len: int = 600,
) -> DualPathInversionBridge:
    stead = load_inversion_bridge_from_checkpoint(
        backbone, stead_head, device,
        embed_dim=embed_dim, n_layers=n_layers, infer_seq_len=infer_seq_len,
    )
    synth = load_inversion_bridge_from_checkpoint(
        backbone, synth_head, device,
        embed_dim=embed_dim, n_layers=n_layers, infer_seq_len=infer_seq_len,
    )
    return DualPathInversionBridge(
        stead=stead,
        synth=synth,
        stead_head=str(Path(stead_head)),
        synth_head=str(Path(synth_head)),
    )
