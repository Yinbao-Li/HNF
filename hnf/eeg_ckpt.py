# -*- coding: utf-8 -*-
"""Load frozen Domain-II EEG checkpoints (native / aniso)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hnf.eeg_native_model import EEGHNFNativeClassifier


def load_native_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    arch = str(ckpt.get("arch") or a.get("arch") or "eeg_hnf_native")
    sample_rate = int(a.get("sample_rate", 128))
    epoch_sec = float(a.get("epoch_sec", 10.0))
    embed_dim = int(a.get("embed_dim", 64))
    principle = str(a.get("principle", "huygens_fresnel"))
    rhythm_phase = bool(a.get("rhythm_phase", True))
    dropout = float(a.get("dropout", 0.2))
    seq_len = int(round(epoch_sec * sample_rate))
    sd = ckpt["state_dict"]
    use_delta = any(k.startswith("delta.") for k in sd)
    segment_pool = any(k.startswith("pool_theta.") for k in sd)
    head_in = int(sd["head.0.weight"].shape[1]) if "head.0.weight" in sd else -1
    n_branches = 3 if use_delta else 2
    extras = 8
    if head_in == embed_dim * n_branches + extras + 6:
        include_region = True
    elif head_in == embed_dim * n_branches + extras:
        include_region = False
    else:
        include_region = True
    if any(k.startswith("spatial.diff_L") for k in sd):
        principle = "aniso_diffusion"
    model = EEGHNFNativeClassifier(
        n_channels=19,
        seq_len=seq_len,
        sample_rate=sample_rate,
        embed_dim=embed_dim,
        num_classes=3,
        dropout=dropout,
        principle=principle,
        use_spatial=not bool(a.get("no_spatial", False)),
        use_delta=use_delta,
        segment_pool=segment_pool,
        include_region_in_head=include_region,
        rhythm_phase=rhythm_phase,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, a, arch
