# -*- coding: utf-8 -*-
"""Smoke tests for rheology baselines."""

from __future__ import annotations

import torch

from hnf.rheo_baselines import build_baseline


def test_baselines_forward_shapes():
    B, T, d = 4, 32, 2
    gd = torch.randn(B, T, d)
    for name in ("isotropic", "diagonal", "lstm", "tcn", "fir"):
        m = build_baseline(name, dim=d, n_modes=2)
        out = m(gd, dt=0.05)
        assert out.shape == (B, T, d), (name, out.shape)


def test_baseline_backward():
    m = build_baseline("lstm", dim=2)
    gd = torch.randn(2, 16, 2)
    st = torch.randn(2, 16, 2)
    pred = m(gd, 0.05)
    loss = (pred - st).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())
