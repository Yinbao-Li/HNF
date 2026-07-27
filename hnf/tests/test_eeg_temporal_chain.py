# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import torch

from hnf.eeg_native_model import EEGHNFNativeClassifier
from hnf.eeg_temporal_chain import (
    EEG_CHAIN_SHAPE_NAMES,
    eeg_temporal_chain_features,
    extract_eeg_temporal_observables,
    features_to_vector,
)


def test_eeg_temporal_chain_features_shape():
    model = EEGHNFNativeClassifier(
        n_channels=19,
        seq_len=1280,
        sample_rate=128,
        embed_dim=32,
        num_classes=3,
        use_delta=False,
        segment_pool=False,
        include_region_in_head=False,
    )
    model.eval()
    x = torch.randn(1, 19, 1280)
    obs = extract_eeg_temporal_observables(model, x, epoch_sec=10.0, clinical_group="HC")
    feat = eeg_temporal_chain_features(obs)
    vec = features_to_vector(feat)
    assert vec.shape == (len(EEG_CHAIN_SHAPE_NAMES),)
    assert np.isfinite(vec).all()
    assert "theta_alpha_lag_norm" in feat
