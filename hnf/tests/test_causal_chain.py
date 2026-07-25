# -*- coding: utf-8 -*-
"""The causal frame must factor hypocentral distance out of chain shape."""

from __future__ import annotations

import numpy as np
import pytest

from hnf.causal_chain import (
    CAUSAL_SHAPE_NAMES,
    CausalObservables,
    causal_chain_features,
    features_to_vector,
    has_valid_chain,
)


def _synthetic_chain(gap_sec: float, window_sec: float = 60.0, seq: int = 800) -> CausalObservables:
    """Same physics (shapes tied to P/S), only the P-S gap differs."""
    t = np.linspace(0.0, window_sec, seq)
    p_sec = 5.0
    s_sec = p_sec + gap_sec

    def bump(center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - center) / width) ** 2)

    # widths scale with the gap so the *shape* in tau is gap-independent
    p_env = bump(p_sec, 0.08 * gap_sec)
    s_env = 0.6 * bump(s_sec, 0.12 * gap_sec)
    # coda after S decays at a fixed rate per tau unit
    coda = np.where(t > s_sec, np.exp(-(t - s_sec) / (0.5 * gap_sec)), 0.0)
    s_env = s_env + 0.3 * coda
    rho = 0.2 + 0.8 * bump(p_sec, 0.1 * gap_sec) + 0.5 * bump(s_sec, 0.15 * gap_sec)

    p_prob = bump(p_sec, 0.2)
    s_prob = bump(s_sec, 0.3)
    wave_env = p_env + s_env + 0.02
    return CausalObservables(
        rho=rho,
        p_env=p_env,
        s_env=s_env,
        p_prob=p_prob,
        s_prob=s_prob,
        wave_env=wave_env,
        det=0.99,
        p_sec=p_sec,
        s_sec=s_sec,
        ps_gap_sec=gap_sec,
        window_sec=window_sec,
        is_event=True,
    )


def test_features_are_finite_and_right_length() -> None:
    obs = _synthetic_chain(gap_sec=6.0)
    vec = features_to_vector(causal_chain_features(obs))
    assert vec.shape == (len(CAUSAL_SHAPE_NAMES),)
    assert np.isfinite(vec).all()


@pytest.mark.parametrize("gap_a,gap_b", [(3.0, 12.0), (4.0, 20.0)])
def test_same_physics_different_distance_maps_close(gap_a: float, gap_b: float) -> None:
    """Two identical-shape chains at very different gaps should look alike."""
    va = features_to_vector(causal_chain_features(_synthetic_chain(gap_a)))
    vb = features_to_vector(causal_chain_features(_synthetic_chain(gap_b)))
    # distance between the two normalised chains must be small relative to scale
    denom = np.linalg.norm(va) + np.linalg.norm(vb) + 1e-8
    rel = np.linalg.norm(va - vb) / denom
    assert rel < 0.15, f"causal frame failed to remove distance: rel={rel:.3f}"


def test_no_chain_without_s() -> None:
    obs = _synthetic_chain(gap_sec=6.0)
    obs.s_sec = -1.0
    obs.ps_gap_sec = -1.0
    assert not has_valid_chain(obs)
    with pytest.raises(ValueError):
        causal_chain_features(obs)


def test_noise_has_no_chain() -> None:
    obs = _synthetic_chain(gap_sec=6.0)
    obs.is_event = False
    assert not has_valid_chain(obs)


def test_multipath_raises_peak_count() -> None:
    simple = _synthetic_chain(gap_sec=8.0)
    multi = _synthetic_chain(gap_sec=8.0)
    t = np.linspace(0.0, multi.window_sec, multi.rho.size)
    # inject a reflector between P and S
    mid = 0.5 * (multi.p_sec + multi.s_sec)
    multi.rho = multi.rho + 0.6 * np.exp(-0.5 * ((t - mid) / 0.3) ** 2)
    n_simple = causal_chain_features(simple)["n_rho_peaks"]
    n_multi = causal_chain_features(multi)["n_rho_peaks"]
    assert n_multi > n_simple
