# -*- coding: utf-8 -*-
"""Tests for Prony / Boltzmann rheology memory."""

from __future__ import annotations

import torch

from hnf.rheo_dataset import RheoMemoryDataset, RheoMemoryModel, rheo_memory_loss
from hnf.rheo_memory import PronyBoltzmannKernel, boltzmann_convolution_dense
from hnf.rheo_synth import make_rheo_sample


def test_prony_positive_params():
    k = PronyBoltzmannKernel(n_modes=3, dim=1, lambda_init=[0.1, 1.0, 10.0], g_init=[0.5, 1.0, 0.2])
    assert torch.all(k.relaxation_times() > 0)
    assert torch.all(k.modal_weights() > 0)
    assert k.g_inf() >= 0


def test_recurrence_matches_dense_isotropic():
    k = PronyBoltzmannKernel(
        n_modes=2, dim=1, lambda_init=[0.5, 2.0], g_init=[1.0, 0.5], g_inf_init=0.1
    )
    k.eval()
    gd = torch.linspace(0.1, 0.3, 48).unsqueeze(0)  # (1, T)
    dt = 0.05
    with torch.no_grad():
        fast = k(gd, dt=dt)
        dense = boltzmann_convolution_dense(gd, k, dt=dt)
    # Recurrence is exact ODE integration; dense is rectangle-rule convolution —
    # they should be close but not bit-identical.
    rel = (fast - dense).pow(2).sum().sqrt() / dense.pow(2).sum().sqrt().clamp_min(1e-8)
    assert float(rel) < 0.15


def test_gt_sample_finite():
    s = make_rheo_sample(n_steps=64, dt=0.05, n_modes=2, seed=0, noise_std=0.0)
    assert s["gammadot"].shape == (64,)
    assert s["stress"].shape == (64,)
    assert s["lambda"].shape == (2,)
    assert abs(float(s["stress"].mean())) > 0 or abs(float(s["stress"].std())) > 0


def test_anisotropic_forward_shape():
    k = PronyBoltzmannKernel(
        n_modes=2, dim=2, anisotropic=True, lambda_init=[0.8, 3.0], g_init=[1.0, 0.5]
    )
    gd = torch.randn(4, 32, 2)
    out = k(gd, dt=0.05)
    assert out.shape == (4, 32, 2)


def test_model_loss_backward():
    model = RheoMemoryModel(n_modes=2, dim=1)
    ds = RheoMemoryDataset("train", n_samples=4, n_steps=64, n_modes=2, seed=1)
    batch_gd = torch.stack([ds[i]["gammadot"] for i in range(4)])
    batch_st = torch.stack([ds[i]["stress"] for i in range(4)])
    dt = 0.05
    pred = model(batch_gd, dt)
    loss, stats = rheo_memory_loss(pred, batch_st, param_reg=1e-4, kernel=model.kernel)
    loss.backward()
    assert stats["mse"] >= 0
    assert model.kernel.raw_lambda.grad is not None


def test_perfect_kernel_near_zero_error():
    """If model params = GT params, stress error ≈ noise floor."""
    s = make_rheo_sample(
        n_steps=128,
        dt=0.05,
        n_modes=2,
        seed=7,
        noise_std=0.0,
        protocol="startup_shear",
        lambdas=[0.5, 5.0],
        weights=[1.2, 0.6],
        g_inf=0.0,
    )
    model = RheoMemoryModel(
        n_modes=2,
        dim=1,
        lambda_init=s["lambda"].tolist(),
        g_init=s["G"].tolist(),
    )
    with torch.no_grad():
        model.kernel.raw_G_inf.fill_(-12.0)
        pred = model(torch.from_numpy(s["gammadot"]).unsqueeze(0), float(s["dt"]))
        gt = torch.from_numpy(s["stress"]).unsqueeze(0)
        rel = (pred - gt).pow(2).sum().sqrt() / gt.pow(2).sum().sqrt().clamp_min(1e-8)
    assert float(rel) < 0.05
