# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from hnf.eeg_braindecode_models import build_braindecode_model


def test_braindecode_models_forward():
    x = torch.randn(2, 19, 1280)
    for name in ("eegnetv4", "shallowfbcsp", "deep4net", "eegconformer"):
        m = build_braindecode_model(name, n_channels=19, n_samples=1280, n_classes=3, dropout=0.35)
        y = m(x)
        assert y.shape == (2, 3), f"{name}: {tuple(y.shape)}"
