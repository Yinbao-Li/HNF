# -*- coding: utf-8 -*-
"""Tests for STEAD picking dataset augmentation."""

from __future__ import annotations

import numpy as np
import torch

from hnf.stead_picking_dataset import STEADPickingDataset


def test_apply_augment_broadcast():
    ds = STEADPickingDataset.__new__(STEADPickingDataset)
    ds.augment = True
    ds.aug_channel_scale = (0.8, 1.2)
    ds.aug_amp_scale = (0.5, 2.0)
    ds.aug_noise_snr_db = (5.0, 20.0)
    ds.aug_time_shift_sec = 0.05
    ds.seq_len = 800

    x = torch.randn(800, 3)
    rng = np.random.default_rng(0)
    out_x, p_idx, s_idx, p_valid, s_valid = ds._apply_augment(x, 100, 200, 1.0, 1.0, rng)
    assert out_x.shape == (800, 3)
