# -*- coding: utf-8 -*-
"""Tests for HNF field reconstruction (user's pipeline)."""

import torch

from hnf.data_generator import build_synthetic_sample
from hnf.field import HuygensNeuralField, solve_weights


def test_solve_weights_shape():
    n = 10
    k = torch.randn(n, n)
    k = k @ k.T + 0.1 * torch.eye(n)
    y = torch.randn(n, 1)
    w = solve_weights(k, y, alpha=1e-2)
    assert w.shape == (n, 1)


def test_positive_definite_with_regularization():
    sample = build_synthetic_sample(n_obs=32, resolution=16, seed=0)
    model = HuygensNeuralField(alpha=1e-2, eps=1e-2, learnable_gamma=False, learnable_omega=False)
    k_obs = model.kernel.forward_cross(
        sample.obs_coords.unsqueeze(0),
        sample.obs_coords.unsqueeze(0),
        return_complex=True,
    ).real.squeeze(0)
    n = k_obs.shape[0]
    system = k_obs + model.alpha * torch.eye(n)
    eigvals = torch.linalg.eigvalsh(system)
    assert (eigvals > 0).all()


def test_interpolation_near_observations():
    sample = build_synthetic_sample(field_type="plane_wave", n_obs=64, resolution=32, seed=1)
    model = HuygensNeuralField(gamma=0.3, omega=4.0, alpha=1e-2, eps=1e-2, learnable_gamma=False, learnable_omega=False)
    pred_at_obs = model.fit_at_observations(sample.obs_coords, sample.obs_values)
    rel_err = (pred_at_obs - sample.obs_values).abs().mean() / sample.obs_values.abs().mean()
    assert rel_err < 0.15


def test_forward_output_shape():
    sample = build_synthetic_sample(n_obs=40, resolution=24, seed=2)
    model = HuygensNeuralField(eps=1e-2)
    out = model(sample.obs_coords, sample.obs_values, sample.grid_coords)
    assert out.shape == sample.field_values.shape


def test_gpu_forward_if_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    sample = build_synthetic_sample(n_obs=32, resolution=16, seed=3)
    model = HuygensNeuralField(eps=1e-2).to(device)
    out = model(
        sample.obs_coords.to(device),
        sample.obs_values.to(device),
        sample.grid_coords.to(device),
    )
    assert out.device.type == "cuda"
    assert out.shape == sample.field_values.shape


def test_gradient_flow_through_learnable_params():
    sample = build_synthetic_sample(n_obs=24, resolution=16, seed=4)
    model = HuygensNeuralField(learnable_gamma=True, learnable_omega=True, eps=1e-2)
    pred = model(sample.obs_coords, sample.obs_values, sample.grid_coords)
    loss = ((pred - sample.field_values) ** 2).mean()
    loss.backward()
    assert model.kernel.gamma.grad is not None
    assert model.kernel.omega.grad is not None
