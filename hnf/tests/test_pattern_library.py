# -*- coding: utf-8 -*-
"""Unit tests for pattern library (no GPU)."""

from __future__ import annotations

import numpy as np
import torch

from hnf.pattern_library import (
    FEATURE_NAMES,
    PatternLibrary,
    PatternPolicy,
    apply_route_crop,
    crop_trace_around_sec,
    features_to_vector,
    waveform_energy_features,
)


def test_kmeans_library_build_and_route():
    rng = np.random.default_rng(0)
    # two blobs: noise-like (low det) vs event-like (high det, gap~3s)
    noise = np.zeros((40, len(FEATURE_NAMES)))
    noise[:, FEATURE_NAMES.index("det")] = rng.normal(0.1, 0.05, 40)
    noise[:, FEATURE_NAMES.index("p_peak")] = rng.normal(0.1, 0.05, 40)
    noise[:, FEATURE_NAMES.index("ps_gap_sec")] = -1.0

    event = np.zeros((40, len(FEATURE_NAMES)))
    event[:, FEATURE_NAMES.index("det")] = rng.normal(0.9, 0.05, 40)
    event[:, FEATURE_NAMES.index("p_peak")] = rng.normal(0.7, 0.05, 40)
    event[:, FEATURE_NAMES.index("s_peak")] = rng.normal(0.6, 0.05, 40)
    event[:, FEATURE_NAMES.index("p_sec")] = 8.0
    event[:, FEATURE_NAMES.index("s_sec")] = 11.0
    event[:, FEATURE_NAMES.index("ps_gap_sec")] = 3.0

    feats = np.vstack([noise, event])
    is_event = np.array([0] * 40 + [1] * 40)
    lib = PatternLibrary.build_from_feature_matrix(feats, is_event=is_event, k=2, seed=0)
    assert len(lib.prototypes) == 2
    # route a clear noise vector
    feat = {n: 0.0 for n in FEATURE_NAMES}
    feat["det"] = 0.05
    feat["p_peak"] = 0.05
    feat["ps_gap_sec"] = -1.0
    d = lib.route(feat)
    assert d.policy.skip_pick or "noise" in d.policy.name or d.policy.name == "noise_skip"


def test_event_majority_cluster_never_noise_skip():
    """Collapsed coarse det/P must not skip an event-dominated cluster."""
    from hnf.pattern_library import _policy_from_cluster_stats

    pol = _policy_from_cluster_stats(
        n_event=1113,
        n_noise=8,
        mean_gap=-1.0,
        mean_det=0.05,
        mean_p_peak=0.05,
    )
    assert not pol.skip_pick
    assert pol.name != "noise_skip"


def test_feedback_ema_updates_center():
    center = [0.0] * len(FEATURE_NAMES)
    center[FEATURE_NAMES.index("det")] = 0.5
    from hnf.pattern_library import PatternPrototype

    proto = PatternPrototype(pattern_id=0, name="t", center=center, count=10)
    lib = PatternLibrary([proto], mean=np.zeros(len(FEATURE_NAMES)), std=np.ones(len(FEATURE_NAMES)))
    feat = {n: 0.0 for n in FEATURE_NAMES}
    feat["det"] = 1.0
    # default: counters only
    lib.update_from_fine(0, feat, confirmed=True, ema=0.5)
    assert abs(lib.prototypes[0].center[FEATURE_NAMES.index("det")] - 0.5) < 1e-6
    assert lib.prototypes[0].n_confirm == 1
    # explicit centre update stays opt-in
    lib.update_from_fine(0, feat, confirmed=True, ema=0.5, update_center=True)
    assert abs(lib.prototypes[0].center[FEATURE_NAMES.index("det")] - 0.75) < 1e-6


def test_crop_and_route_shapes():
    x = torch.randn(1, 800, 3)
    t = torch.linspace(0, 60, 800).unsqueeze(-1)
    x2, t2, shift = crop_trace_around_sec(x, t, 8.0, half_sec=6.0, window_sec=60.0, out_len=800)
    assert x2.shape == (1, 800, 3)
    assert t2.shape == (800, 1)
    assert shift >= 0
    pol = PatternPolicy(name="near_ps_crop_p", crop_around="p", crop_half_sec=6.0)
    feat = {n: 0.0 for n in FEATURE_NAMES}
    feat["p_sec"] = 8.0
    xo, to, sh = apply_route_crop(x, t, feat, pol, window_sec=60.0, out_len=800)
    assert xo.shape == (1, 800, 3)


def test_energy_features_finite():
    x = torch.randn(200, 3)
    a, b, r = waveform_energy_features(x)
    assert np.isfinite([a, b, r]).all()


def test_save_load_roundtrip(tmp_path):
    feats = np.random.randn(30, len(FEATURE_NAMES))
    feats[:, FEATURE_NAMES.index("det")] = np.linspace(0, 1, 30)
    lib = PatternLibrary.build_from_feature_matrix(feats, k=3, seed=1)
    path = tmp_path / "lib.json"
    lib.save(path)
    lib2 = PatternLibrary.load(path)
    assert len(lib2.prototypes) == len(lib.prototypes)
    v = features_to_vector({n: 0.1 for n in FEATURE_NAMES})
    assert lib2.route({n: 0.1 for n in FEATURE_NAMES}).pattern_id >= 0
