# -*- coding: utf-8 -*-
"""Tests for spatial fluid HNF."""

from __future__ import annotations

import torch

from hnf.fluid_spatial import (
    SpatialFluidHNFReconstructor,
    SpatialRotatingHuygensBlock,
    build_spatial_stencil,
    curl_2d,
    grid_coords,
)


def test_grid_coords_shape():
    c = grid_coords(16, 20)
    assert c.shape == (2, 16, 20)


def test_stencil_rotational_kernel_center_safe():
    st = build_spatial_stencil(5, cell_size=0.1)
    center = st.kernel_size // 2
    idx = center * st.kernel_size + center
    assert torch.isfinite(st.rot_x[idx])
    assert torch.isfinite(st.rot_y[idx])


def test_spatial_block_forward():
    b = SpatialRotatingHuygensBlock(channels=32, kernel_size=5, use_rotation=True)
    feat = torch.randn(2, 32, 16, 16)
    rho = torch.ones(2, 1, 16, 16)
    out = b(feat, rho)
    assert out.shape == feat.shape
    assert torch.isfinite(out).all()


def test_spatial_model_forward_and_rotation_ablation():
    x = torch.randn(2, 3, 32, 32)
    m_rot = SpatialFluidHNFReconstructor(h=32, w=32, use_rotation=True, predict_eta=True)
    m_lin = SpatialFluidHNFReconstructor(h=32, w=32, use_rotation=False, predict_eta=False)
    pred, aux = m_rot(x, return_aux=True)
    assert pred.shape == (2, 2, 32, 32)
    assert aux["rho"].shape == (2, 1, 32, 32)
    assert aux["eta"].shape == (2,)
    pred2 = m_lin(x)
    assert pred2.shape == (2, 2, 32, 32)


def test_curl_2d_vortex_sign():
    # Solid-body rotation v = (-y, x) has curl = 2
    h, w = 32, 32
    c = grid_coords(h, w)
    xx, yy = c[0], c[1]
    v = torch.stack([-yy, xx], dim=0).unsqueeze(0)
    w_field = curl_2d(v)
    assert w_field.mean().item() > 0.0
