# -*- coding: utf-8 -*-
"""Tests for banded sparse Huygens kernel."""

from __future__ import annotations

import torch

from hnf.kernel import HuygensKernel
from hnf.layers import HuygensWaveBlock


def test_sparse_matches_dense_small():
    torch.manual_seed(0)
    b, n, d = 2, 64, 16
    h_real = torch.randn(b, n, d)
    h_imag = torch.randn(b, n, d) * 0.1
    t = torch.linspace(0, 60, n).view(1, n, 1).expand(b, -1, -1)
    rho = torch.ones(b, n, 1)

    block_dense = HuygensWaveBlock(
        dim=d,
        local_window_sec=15.0,
        distance_mode="time",
        sparse_band=False,
    )
    block_sparse = HuygensWaveBlock(
        dim=d,
        local_window_sec=15.0,
        distance_mode="time",
        sparse_band=True,
    )
    block_sparse.load_state_dict(block_dense.state_dict())

    out_d = block_dense(h_real.clone(), h_imag.clone(), t=t, rho=rho)
    out_s = block_sparse(h_real.clone(), h_imag.clone(), t=t, rho=rho)

    for od, os_ in zip(out_d, out_s):
        assert torch.allclose(od, os_, atol=1e-4, rtol=1e-3)


def test_forward_apply_vs_dense():
    b, n, d = 1, 80, 8
    h_c = torch.complex(torch.randn(b, n, d), torch.randn(b, n, d) * 0.05)
    x = torch.randn(b, n, d)
    t = torch.linspace(0, 60, n).view(1, n, 1)
    rho = torch.ones(b, n, 1)

    k_dense = HuygensKernel(
        local_window_sec=15.0,
        distance_mode="time",
        sparse_band=False,
    )
    k_sparse = HuygensKernel(
        local_window_sec=15.0,
        distance_mode="time",
        sparse_band=True,
    )
    out_dense = torch.matmul(k_dense(x, t=t, rho=rho, return_complex=True), h_c)
    out_sparse = k_sparse.forward_apply(h_c, x, t=t, rho=rho)
    assert torch.allclose(out_dense.real, out_sparse.real, atol=1e-4, rtol=1e-3)
    assert torch.allclose(out_dense.imag, out_sparse.imag, atol=1e-4, rtol=1e-3)
