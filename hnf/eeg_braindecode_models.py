# -*- coding: utf-8 -*-
"""Braindecode SOTA EEG classifiers (official implementations)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from braindecode.models import Deep4Net, EEGConformer, EEGNetv4, ShallowFBCSPNet

SOTA_CITATIONS: dict[str, str] = {
    "eegnetv4": "Lawhern et al., J. Neural Eng. 2018 (Braindecode EEGNetv4)",
    "shallowfbcsp": "Schirrmeister et al., Hum. Brain Mapp. 2017 (Braindecode ShallowFBCSPNet)",
    "deep4net": "Schirrmeister et al., Hum. Brain Mapp. 2017 (Braindecode Deep4Net)",
    "eegconformer": "Song et al., IEEE TNSRE 2023 (Braindecode EEGConformer)",
}


def build_braindecode_model(
    name: str,
    *,
    n_channels: int = 19,
    n_samples: int = 1280,
    n_classes: int = 3,
    dropout: float = 0.25,
    extra: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a Braindecode model for (B, C, T) EEG epochs."""
    key = name.strip().lower().replace("-", "").replace("_", "")
    kw = dict(extra or {})
    common = dict(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_samples,
    )

    if key in {"eegnet", "eegnetv4"}:
        return EEGNetv4(**common, drop_prob=dropout, **kw)
    if key in {"shallow", "shallowfbcsp", "shallowfbcspnet"}:
        return ShallowFBCSPNet(
            **common,
            drop_prob=dropout,
            final_conv_length="auto",
            add_log_softmax=False,
            **kw,
        )
    if key in {"deep4", "deep4net", "deep"}:
        return Deep4Net(**common, drop_prob=dropout, add_log_softmax=False, **kw)
    if key in {"conformer", "eegconformer"}:
        att_drop = float(kw.pop("att_drop_prob", max(dropout, 0.35)))
        return EEGConformer(
            **common,
            drop_prob=dropout,
            att_drop_prob=att_drop,
            final_fc_length="auto",
            add_log_softmax=False,
            **kw,
        )
    raise ValueError(f"Unknown Braindecode model: {name!r}")


def display_name(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    return {
        "eegnet": "EEGNetv4 (Braindecode)",
        "eegnetv4": "EEGNetv4 (Braindecode)",
        "shallow": "ShallowFBCSPNet (Braindecode)",
        "shallowfbcsp": "ShallowFBCSPNet (Braindecode)",
        "shallowfbcspnet": "ShallowFBCSPNet (Braindecode)",
        "deep4": "Deep4Net (Braindecode)",
        "deep4net": "Deep4Net (Braindecode)",
        "deep": "Deep4Net (Braindecode)",
        "conformer": "EEG Conformer (Braindecode)",
        "eegconformer": "EEG Conformer (Braindecode)",
    }.get(key, name)
