# -*- coding: utf-8 -*-
"""Tests for fluid reconstruction losses."""

from __future__ import annotations

import torch

from hnf.fluid_losses import (
    curl_weight_at_epoch,
    normalized_curl_loss,
    velocity_recon_loss,
)


def test_velocity_recon_loss_emphasizes_missing():
    pred = torch.zeros(1, 2, 4, 4)
    gt = torch.zeros(1, 2, 4, 4)
    gt[0, :, 1:, :] = 2.0  # missing region (non-observed) has larger error
    obs = torch.zeros(1, 1, 4, 4)
    obs[0, 0, 0, 0] = 1.0
    loss_high_recon, _ = velocity_recon_loss(pred, gt, obs, obs_weight=0.1, recon_weight=1.0)
    loss_low_recon, _ = velocity_recon_loss(pred, gt, obs, obs_weight=1.0, recon_weight=0.1)
    assert loss_high_recon.item() > loss_low_recon.item()


def test_curl_weight_warmup():
    assert curl_weight_at_epoch(0, 0.01, 10) == 0.0
    assert curl_weight_at_epoch(5, 0.01, 10) == 0.005
    assert curl_weight_at_epoch(10, 0.01, 10) == 0.01


def test_normalized_curl_zero_on_match():
    gt = torch.randn(1, 2, 16, 16)
    loss = normalized_curl_loss(gt, gt)
    assert loss.item() < 1e-5
