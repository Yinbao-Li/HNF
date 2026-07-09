# -*- coding: utf-8 -*-
"""Tests for Huygens kernel properties (user's HuygensKernel API)."""

import torch

from hnf.kernel import HuygensKernel


def test_kernel_real_part_symmetry():
    kernel = HuygensKernel(gamma=0.5, omega=3.0, causal=False)
    coords = torch.randn(12, 2)
    k_real = kernel(coords.unsqueeze(0)).real.squeeze(0)
    assert torch.allclose(k_real, k_real.T, atol=1e-4)


def test_kernel_diagonal_finite():
    kernel = HuygensKernel(eps=1e-2, causal=False)
    coords = torch.randn(8, 3)
    k = kernel(coords.unsqueeze(0)).squeeze(0)
    diag = k.diagonal()
    assert torch.isfinite(diag.real).all()
    assert torch.isfinite(diag.imag).all()


def test_causal_kernel_runs():
    """Causal kernel forward should complete without error."""
    n = 6
    t = torch.linspace(0, 1, n).reshape(1, n, 1)
    coords = torch.stack([torch.zeros(n), t.squeeze()], dim=-1).unsqueeze(0)
    kernel = HuygensKernel(causal=True, wave_speed=1.0)
    k = kernel(coords, t=t, return_complex=False)
    assert k.shape[0] == 1
    assert torch.isfinite(k).all()


def test_learnable_parameters():
    kernel = HuygensKernel(learnable_gamma=True, learnable_omega=True)
    assert kernel.gamma.requires_grad
    assert kernel.omega.requires_grad


def test_forward_cross_shape():
    kernel = HuygensKernel(causal=False)
    a = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[0.5, 0.5]])
    k = kernel.forward_cross(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    assert k.shape == (2, 1)


def test_huygens_fresnel_differs_from_huygens():
    n = 8
    t = torch.linspace(0, 1, n).reshape(1, n, 1)
    x = torch.randn(1, n, 4)
    k_h = HuygensKernel(principle="huygens", causal=True, distance_mode="time", wave_speed=1.0)
    k_f = HuygensKernel(
        principle="huygens_fresnel",
        causal=True,
        distance_mode="time",
        wave_speed=1.0,
        obliquity_scale=1.0,
    )
    kh = k_h(x, t=t, return_complex=True)
    kf = k_f(x, t=t, return_complex=True)
    assert kh.shape == kf.shape
    assert torch.isfinite(kf.real).all() and torch.isfinite(kf.imag).all()
    assert not torch.allclose(torch.abs(kh), torch.abs(kf), rtol=1e-3, atol=1e-4)


def test_fresnel_obliquity_in_unit_interval():
    n = 10
    t = torch.linspace(0, 2, n).reshape(1, n, 1)
    x = torch.randn(1, n, 3)
    kernel = HuygensKernel(principle="huygens_fresnel", causal=True, distance_mode="time")
    r = kernel.resolve_distance(x, t=t)
    chi = kernel._fresnel_obliquity(r, t=t, x=x)
    assert torch.all(chi >= 0.0 - 1e-5)
    assert torch.all(chi <= 1.0 + 1e-5)
