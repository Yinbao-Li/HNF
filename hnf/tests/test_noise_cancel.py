# -*- coding: utf-8 -*-
"""Tests for Huygens noise cancellation branch."""

from __future__ import annotations

import torch

from hnf.noise_cancel import HuygensNoiseCancelBranch, noise_cancel_losses
from hnf.picking_model import build_picking_model


def test_noise_cancel_branch_shapes():
    branch = HuygensNoiseCancelBranch(channels=3, source_dim=8, hidden=16)
    x = torch.randn(2, 64, 3)
    t = torch.linspace(0, 60, 64).view(1, 64, 1).expand(2, -1, -1)
    out = branch(x, t)
    assert out["s_noise"].shape == (2, 64, 8)
    assert out["n_sim"].shape == (2, 64, 3)
    assert out["u_final"].shape == (2, 64, 3)
    assert (out["s_noise"] >= 0).all()


def test_noise_cancel_losses_finite():
    x = torch.randn(2, 64, 3)
    t = torch.linspace(0, 60, 64).view(1, 64, 1).expand(2, -1, -1)
    branch = HuygensNoiseCancelBranch(channels=3, source_dim=8, hidden=16)
    nc_out = branch(x, t)
    batch = {
        "x": x,
        "det": torch.tensor([1.0, 0.0]),
        "p_target": torch.zeros(2, 64),
        "s_target": torch.zeros(2, 64),
    }
    loss, parts = noise_cancel_losses({}, nc_out, batch)
    assert torch.isfinite(loss)
    assert "nc_recon" in parts


def test_picking_model_with_noise_cancel():
    model = build_picking_model(
        embed_dim=32,
        num_branch_layers=1,
        enhanced_det_head=True,
        noise_cancel=True,
        noise_source_dim=8,
    )
    x = torch.randn(2, 64, 3)
    t = torch.linspace(0, 60, 64).view(1, 64, 1).expand(2, -1, -1)
    out = model(x, t)
    assert out["det"].shape == (2,)
    assert out["p"].shape == (2, 64)
    assert "nc_u_final" in out
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000
