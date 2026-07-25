# -*- coding: utf-8 -*-
"""Tests for temporal sampler (linear + learnable)."""

from __future__ import annotations

import torch

from hnf.learnable_sampler import LearnableTemporalSampler, remap_index, remap_sequence


def test_linear_sampler_shapes_and_finite():
    samp = LearnableTemporalSampler(channels=3, out_len=80, mode="linear")
    x = torch.randn(2, 600, 3)
    out = samp(x)
    assert out["x"].shape == (2, 80, 3)
    assert out["t"].shape == (2, 80, 1)
    assert out["w"].shape == (2, 600)
    assert out["attn"].shape == (2, 600, 80)
    assert torch.isfinite(out["x"]).all()
    # columns of attn sum ~1
    col = out["attn"].sum(dim=1)
    assert torch.allclose(col, torch.ones_like(col), atol=1e-4)


def test_linear_sampler_remap_index():
    samp = LearnableTemporalSampler(channels=3, out_len=100, mode="linear")
    x = torch.randn(1, 1000, 3)
    attn = samp(x)["attn"]
    # Indices that are NOT interpolation knots must still map proportionally.
    idx = torch.tensor([0, 400, 999])
    attn3 = attn.expand(3, -1, -1)
    out = remap_index(attn3, idx)
    assert out[0].item() == 0
    assert out[2].item() == 99
    assert abs(out[1].item() - 40) <= 1


def test_linear_remap_index_matches_time_scale():
    samp = LearnableTemporalSampler(channels=3, out_len=800, mode="linear")
    attn = samp(torch.randn(1, 6000, 3))["attn"]
    for sec in (8.0, 12.0, 20.0):
        fine = int(round(sec / 60.0 * 5999))
        got = int(remap_index(attn.expand(1, -1, -1), torch.tensor([fine]))[0])
        exp = int(round(sec / 60.0 * 799))
        assert abs(got - exp) <= 1, (sec, got, exp)


def test_learnable_uniform_init_finite():
    samp = LearnableTemporalSampler(
        channels=3, out_len=80, temperature=0.25, mode="learnable"
    )
    x = torch.randn(2, 600, 3)
    out = samp(x)
    assert torch.isfinite(out["x"]).all()
    assert torch.isfinite(out["attn"]).all()
    rem = remap_sequence(out["attn"], torch.randn(2, 600))
    assert rem.shape == (2, 80)
    assert torch.isfinite(rem).all()
