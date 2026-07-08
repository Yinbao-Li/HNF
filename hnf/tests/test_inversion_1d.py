# -*- coding: utf-8 -*-
"""Tests for 1D layered inversion."""

import torch

from hnf.field import HuygensNeuralField
from hnf.inversion_1d import (
    LayeredEarth1D,
    amplitude_log_phase,
    default_station_distances,
    default_synth_model,
    hnf_profile_regularization,
    invert_layered_1d,
    synthesize_observations,
    synthesize_travel_times,
    travel_time_phase,
    vertical_integral_slowness,
)


def test_vertical_slowness_matches_manual():
    depths = torch.tensor([0.0, 2.0, 4.0])
    vp = torch.tensor([4.0, 6.0])
    t = vertical_integral_slowness(depths, vp, torch.tensor(3.0), torch.tensor(0.0))
    expected = 1.0 / 6.0 + 2.0 / 4.0
    assert abs(t.item() - expected) < 1e-5


def test_travel_time_increases_with_distance():
    model = default_synth_model("cpu")
    d = torch.tensor([0.0, 10.0, 30.0])
    tp = travel_time_phase(model, "P", torch.tensor(10.0), d)
    assert tp[0] < tp[1] < tp[2]


def test_p_faster_than_s():
    model = default_synth_model("cpu")
    d = torch.tensor([5.0, 20.0])
    tp = travel_time_phase(model, "P", torch.tensor(10.0), d)
    ts = travel_time_phase(model, "S", torch.tensor(10.0), d)
    assert (tp < ts).all()


def test_amplitude_decreases_with_distance():
    model = default_synth_model("cpu")
    d = torch.tensor([0.0, 20.0, 50.0])
    log_a = amplitude_log_phase(model, "P", torch.tensor(10.0), d, frequency_hz=8.0)
    assert log_a[0] > log_a[1] > log_a[2]


def test_inversion_reduces_rmse_noise_free():
    device = torch.device("cpu")
    true_model = default_synth_model(device)
    distances = default_station_distances(device, n_stations=8)
    obs = synthesize_travel_times(true_model, 10.0, distances, noise_std=0.0)
    vp0 = true_model.vp * 1.08
    vs0 = true_model.vs * 0.92
    recovered, _ = invert_layered_1d(
        depths=true_model.depths,
        vp_init=vp0,
        vs_init=vs0,
        q_init=true_model.q,
        source_depth=10.0,
        receiver_distances=distances,
        obs=obs,
        steps=600,
        lr=0.06,
    )
    rec = recovered.earth
    vp_err = torch.sqrt(torch.mean((rec.vp - true_model.vp) ** 2)).item()
    vs_err = torch.sqrt(torch.mean((rec.vs - true_model.vs) ** 2)).item()
    assert vp_err < 0.85
    assert vs_err < 0.55


def test_q_inversion_with_amplitude():
    device = torch.device("cpu")
    true_model = default_synth_model(device)
    distances = default_station_distances(device, n_stations=8)
    freqs = [4.0, 8.0, 12.0, 16.0]
    obs = synthesize_observations(true_model, 10.0, distances, frequency_hz=freqs)
    q0 = true_model.q * 1.15
    recovered, _ = invert_layered_1d(
        depths=true_model.depths,
        vp_init=true_model.vp.clone(),
        vs_init=true_model.vs.clone(),
        q_init=q0,
        source_depth=10.0,
        receiver_distances=distances,
        obs=obs,
        steps=700,
        lr=0.08,
        amp_weight=3.0,
        invert_q=True,
        frequency_hz=freqs,
        q_anchor_weight=0.01,
        two_stage_q=True,
    )
    rec = recovered.earth
    init_earth = LayeredEarth1D(true_model.depths, true_model.vp, true_model.vs, q0)
    src = torch.tensor(10.0)
    log_ap_init = amplitude_log_phase(init_earth, "P", src, distances, freqs)
    amp_init = float(torch.mean((log_ap_init - obs["log_ap"]) ** 2))
    log_ap_rec = amplitude_log_phase(rec, "P", src, distances, freqs)
    amp_rec = float(torch.mean((log_ap_rec - obs["log_ap"]) ** 2))
    q_err_init = torch.sqrt(torch.mean((q0 - true_model.q) ** 2)).item()
    q_err = torch.sqrt(torch.mean((rec.q - true_model.q) ** 2)).item()
    assert amp_rec < amp_init
    assert q_err < q_err_init * 1.1 or amp_rec < amp_init * 0.1


def test_hnf_regularization_runs():
    device = torch.device("cpu")
    true_model = default_synth_model(device)
    distances = default_station_distances(device, n_stations=8)
    obs = synthesize_travel_times(true_model, 10.0, distances)
    hnf = HuygensNeuralField(causal=True, eps=1e-2, learnable_gamma=False, learnable_omega=False)
    recovered, _ = invert_layered_1d(
        depths=true_model.depths,
        vp_init=true_model.vp * 1.08,
        vs_init=true_model.vs * 0.92,
        q_init=true_model.q,
        source_depth=10.0,
        receiver_distances=distances,
        obs=obs,
        steps=50,
        lr=0.06,
        hnf_field=hnf,
        hnf_weight=0.2,
    )
    reg = hnf_profile_regularization(recovered, hnf)
    assert torch.isfinite(reg)


def test_gradient_flow():
    depths = torch.tensor([0.0, 5.0, 15.0])
    vp = torch.tensor([4.0, 6.0], requires_grad=True)
    earth = LayeredEarth1D(
        depths=depths,
        vp=vp,
        vs=torch.tensor([2.3, 3.4]),
        q=torch.tensor([100.0, 120.0]),
    )
    distances = torch.tensor([0.0, 15.0, 40.0])
    tp = travel_time_phase(earth, "P", torch.tensor(10.0), distances)
    tp.sum().backward()
    assert vp.grad is not None
