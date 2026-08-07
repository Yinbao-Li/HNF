# -*- coding: utf-8 -*-
"""Unit tests for subject-level diffusion fit + residual helpers."""

from __future__ import annotations

import numpy as np

from hnf.eeg_geometry import electrode_xyz
from hnf.eeg_subject_diffusion import (
    atrophy_template_scores,
    diffusion_kernel_from_vec,
    fit_subject_diffusion,
    residualize,
    region_coupling,
)


def test_residualize_recovers_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((80, 2))
    y = 1.5 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.05 * rng.standard_normal(80)
    resid, r2, beta = residualize(y, x)
    assert r2 > 0.95
    assert abs(beta[1] - 2.0) < 0.15
    assert float(np.nanstd(resid)) < 0.2


def test_fit_recovers_anisotropic_kernel_correlation():
    xyz = electrode_xyz().astype(np.float64)
    # Stronger diffusion along +x (electrode left–right).
    vec_true = np.array([0.2, 0.0, -1.5, 0.0, 0.0, -1.5, -0.8], dtype=np.float64)
    K, D_true, _ = diffusion_kernel_from_vec(xyz, vec_true)
    rng = np.random.default_rng(1)
    # Synthesize epochs whose corr ≈ K + noise.
    C = K.copy()
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    w, v = np.linalg.eigh(C + 0.05 * np.eye(19))
    w = np.clip(w, 1e-4, None)
    A = v @ np.diag(np.sqrt(w))
    epochs = []
    for _ in range(24):
        z = rng.standard_normal((19, 256))
        epochs.append(A @ z)
    fit = fit_subject_diffusion(np.stack(epochs), xyz=xyz, maxiter=80)
    assert fit["fit_r"] > 0.4
    assert fit["D_aniso"] > 1.05
    eig_true = np.sort(np.linalg.eigvalsh(D_true))
    # Dominant axis should remain the largest eigenvalue (loose).
    assert fit["D_eig2"] > fit["D_eig0"]
    assert eig_true[-1] > eig_true[0]


def test_atrophy_templates_split_profiles():
    ftd_like = {
        "couple_frontal": 0.05,
        "couple_temporal": 0.08,
        "couple_central": 0.40,
        "couple_posterior": 0.45,
    }
    ad_like = {
        "couple_frontal": 0.40,
        "couple_temporal": 0.10,
        "couple_central": 0.35,
        "couple_posterior": 0.06,
    }
    s_ftd = atrophy_template_scores(ftd_like)
    s_ad = atrophy_template_scores(ad_like)
    assert s_ftd["tmpl_ftd_minus_ad"] > s_ad["tmpl_ftd_minus_ad"]
    coup = region_coupling(np.eye(19))
    assert "couple_frontal" in coup
