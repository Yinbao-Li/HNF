# -*- coding: utf-8 -*-
"""Tests for picking prior and FWI-lite modules."""

import torch

from hnf.acoustic_fwi_1d import DirectWaveForward
from hnf.inversion_1d import default_synth_model, travel_time_phase
from hnf.picking_prior import kernel_ratio_to_vs, rho_to_q_prior
from hnf.synth_waveforms_1d import synthesize_multistation_batch


def test_synth_multistation_shapes():
    model = default_synth_model("cpu")
    d = torch.linspace(0, 40, 4)
    x, t, meta = synthesize_multistation_batch(model, 10.0, d, seq_len=200, noise_std=0.0, seed=1)
    assert x.shape == (4, 200, 3)
    assert t.shape == (200, 1)
    assert meta["tp_sec"].numel() == 4


def test_rho_to_q_prior_monotonic():
    dev = torch.device("cpu")
    rho = torch.rand(4, 100, device=dev) + 0.5
    tp = torch.tensor([1.0, 3.0, 5.0, 8.0], device=dev)
    ts = tp + 2.0
    q_ref = torch.tensor([80.0, 120.0, 150.0, 200.0, 250.0], device=dev)
    q, layers = rho_to_q_prior(rho, tp, ts, n_layers=5, q_ref=q_ref)
    assert q.shape == (5,)
    assert layers.shape == (5,)
    assert (q[1:] >= q[:-1]).all()


def test_kernel_ratio_vs():
    vp = torch.tensor([3.5, 4.5, 5.5, 6.2, 6.8])
    vs = kernel_ratio_to_vs(vp, kernel_vp=8.0, kernel_vs=4.5)
    assert vs.shape == vp.shape
    assert (vs < vp).all()


def test_direct_wave_forward_finite_and_grad():
    model = default_synth_model("cpu")
    engine = DirectWaveForward(device="cpu", nt=200)
    d = torch.tensor([0.0, 15.0, 30.0])
    vp = model.vp.clone().requires_grad_(True)
    earth = type(model)(depths=model.depths, vp=vp, vs=model.vs, q=model.q)
    seis = engine.simulate(earth, 10.0, d)
    assert seis.shape == (3, 200)
    assert torch.isfinite(seis).all()
    seis.sum().backward()
    assert vp.grad is not None
    assert torch.isfinite(vp.grad).all()


def test_direct_wave_moves_with_velocity():
    model = default_synth_model("cpu")
    engine = DirectWaveForward(device="cpu", nt=300)
    d = torch.tensor([20.0])
    slow = type(model)(depths=model.depths, vp=model.vp * 0.85, vs=model.vs, q=model.q)
    fast = model
    src = torch.tensor(10.0)
    tp_slow = travel_time_phase(slow, "P", src, d)
    tp_fast = travel_time_phase(fast, "P", src, d)
    assert tp_slow.item() > tp_fast.item()
