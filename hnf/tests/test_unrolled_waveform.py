# -*- coding: utf-8 -*-

import torch

from hnf.acoustic_fwi_1d import DirectWaveForward, unrolled_waveform_refine
from hnf.inversion_1d import default_station_distances, default_synth_model


def test_unrolled_waveform_refine_is_differentiable():
    earth = default_synth_model("cpu")
    distances = default_station_distances("cpu", 4)
    engine = DirectWaveForward(device="cpu", nt=256, dt=0.02)
    observed = engine.simulate(earth, 10.0, distances)

    vp0 = (earth.vp * 0.95).clone().requires_grad_(True)
    vs0 = (earth.vs * 0.95).clone().requires_grad_(True)
    refined, metrics = unrolled_waveform_refine(
        earth.depths, vp0, vs0, earth.q, 10.0, distances, observed,
        steps=2, step_size=0.02, dt=0.02,
    )
    loss = refined.vp.mean() + metrics["waveform"]
    loss.backward()
    assert vp0.grad is not None
    assert vs0.grad is not None
