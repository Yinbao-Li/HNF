# -*- coding: utf-8 -*-
"""Tests for multi-scale DeepHuygens picking."""

from __future__ import annotations

import torch

from hnf.picking_model import build_picking_model


def test_baseline_forward():
    model = build_picking_model(embed_dim=32, num_shared_layers=1, num_branch_layers=1)
    x = torch.randn(2, 64, 3)
    t = torch.linspace(0, 60, 64).view(1, 64, 1).expand(2, -1, -1)
    out = model(x, t)
    assert out["det"].shape == (2,)
    assert out["p"].shape == (2, 64)
    assert out["s"].shape == (2, 64)


def test_multiscale_forward():
    model = build_picking_model(
        embed_dim=32,
        num_branch_layers=1,
        multi_scale=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000
    x = torch.randn(2, 80, 3)
    t = torch.linspace(0, 60, 80).view(1, 80, 1).expand(2, -1, -1)
    out = model(x, t)
    assert out["det"].shape == (2,)
    assert out["p"].shape == (2, 80)
    assert out["s"].shape == (2, 80)
    assert model.multi_scale_encoder is not None


def test_partial_resume():
    base = build_picking_model(embed_dim=32, num_branch_layers=1, multi_scale=False)
    ms = build_picking_model(embed_dim=32, num_branch_layers=1, multi_scale=True)
    missing, unexpected = ms.load_state_dict(base.state_dict(), strict=False)
    assert any("multi_scale_encoder" in k for k in missing)
    assert any("shared_layers" in k for k in unexpected)
