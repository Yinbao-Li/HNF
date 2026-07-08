# -*- coding: utf-8 -*-
"""Tests for physics-residual det / pick heads."""

from __future__ import annotations

import torch

from hnf.picking_model import ScalarDetHead, WaveFieldPickingHead, build_picking_model


def test_pick_head_envelope_residual():
    head = WaveFieldPickingHead(hidden=16, residual_envelope=True)
    h_real = torch.randn(2, 32, 8)
    h_imag = torch.randn(2, 32, 8) * 0.1
    out = head(h_real, h_imag)
    assert out.shape == (2, 32)

    head_no = WaveFieldPickingHead(hidden=16, residual_envelope=False)
    out_no = head_no(h_real, h_imag)
    assert not torch.allclose(out, out_no)


def test_scalar_det_energy_skip():
    head = ScalarDetHead(embed_dim=16, residual_energy=True)
    wave_e = torch.rand(4, 16)
    total_e = torch.rand(4) + 0.1
    out0 = head(wave_e, total_e)
    assert out0.shape == (4,)

    with torch.no_grad():
        head.energy_weight.fill_(1.0)
    out1 = head(wave_e, total_e)
    assert not torch.allclose(out0, out1)


def test_model_forward_with_residual_heads():
    model = build_picking_model(
        embed_dim=32,
        num_branch_layers=1,
        residual_pick_head=True,
        residual_det_head=True,
    )
    x = torch.randn(2, 64, 3)
    t = torch.linspace(0, 60, 64).view(1, 64, 1).expand(2, -1, -1)
    out = model(x, t)
    assert out["det"].shape == (2,)
    assert out["p"].shape == (2, 64)
