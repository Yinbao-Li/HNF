# -*- coding: utf-8 -*-
"""Unit tests for EEG pattern library (CPU)."""

from __future__ import annotations

import numpy as np

from hnf.eeg_pattern_library import (
    EEG_ROUTER_FEATURES,
    EEGPatternLibrary,
    EEGPatternPolicy,
    apply_eeg_policy,
    evaluate_routed_subjects,
    subject_router_features,
)


def _fake_subjects(n_per: int = 12) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for label, name, center in [
        (0, "HC", (0.8, 0.1, 0.1)),
        (2, "AD", (0.1, 0.2, 0.7)),
        (1, "FTD", (0.1, 0.55, 0.35)),
        (2, "AD2", (0.15, 0.42, 0.43)),  # confused AD/FTD pocket
    ]:
        for i in range(n_per):
            noise = rng.normal(0, 0.03, 3)
            p = np.clip(np.asarray(center) + noise, 1e-3, None)
            p = p / p.sum()
            rows.append(
                {
                    "subject_id": f"{name}_{i}",
                    "label": label,
                    "prob_hc": float(p[0]),
                    "prob_ftd": float(p[1]),
                    "prob_ad": float(p[2]),
                    "rho_mean": float(0.4 + 0.1 * label + rng.normal(0, 0.02)),
                    "rho_std": 0.2,
                    "rho_p90": 0.8,
                    "rho_cv": 0.5,
                    "bp_delta": 0.1 * label,
                    "bp_theta": 0.2 + 0.05 * label,
                    "bp_alpha": 1.0 - 0.1 * label,
                    "bp_beta": 0.3,
                    "theta_alpha_ratio": 0.1 * label,
                    "hnf_theta_energy": 1.0,
                    "hnf_alpha_energy": 1.2 - 0.05 * label,
                    "hnf_theta_alpha_ratio": 0.05 * label,
                    "hnf_delta_energy": 0.5,
                    "region_ft_contrast": 0.1 * (label - 1),
                    "region_pf_contrast": -0.05 * label,
                }
            )
    return rows


def test_subject_router_features_finite():
    s = _fake_subjects(1)[0]
    f = subject_router_features(s)
    assert set(EEG_ROUTER_FEATURES) <= set(f)
    assert all(np.isfinite(list(f.values())))


def test_apply_second_look_tight():
    pol = EEGPatternPolicy(
        name="ad_ftd_second_look",
        ad_ftd_second_look=True,
        confusion_margin=0.20,
        min_disease_mass=0.55,
        hc_keep_margin=1.15,
    )
    # Confused disease → force FTD
    pred, abs_, reason = apply_eeg_policy(np.array([0.1, 0.48, 0.42]), pol)
    assert not abs_ and pred == 1 and reason == "second_look_force"
    # Clear AD gap → defer to head
    pred, abs_, reason = apply_eeg_policy(np.array([0.1, 0.2, 0.7]), pol)
    assert pred == 2 and reason == "second_look_defer_head"
    # Clear HC
    pred, abs_, reason = apply_eeg_policy(np.array([0.7, 0.2, 0.1]), pol)
    assert pred == 0 and reason == "second_look_keep_hc"
    # OOD distance
    pred, abs_, reason = apply_eeg_policy(
        np.array([0.5, 0.3, 0.2]), pol, distance=9.0, max_route_distance=3.0
    )
    assert abs_ and reason == "ood_distance"


def test_build_calibrate_roundtrip(tmp_path):
    subs = _fake_subjects(10)
    train, val = subs[:30], subs[30:40]
    lib = EEGPatternLibrary.build_from_subjects(train, k=4, seed=0, checkpoint="dummy")
    cal = lib.calibrate_distance_gate(val, min_coverage=0.5)
    assert "chosen" in cal
    path = tmp_path / "lib.json"
    lib.save(path)
    lib2 = EEGPatternLibrary.load(path)
    m = evaluate_routed_subjects(lib2, val, online_update=True, update_center=False)
    assert m["n_subjects"] == len(val)
    assert 0.0 <= m["coverage"] <= 1.0
    # counters moved, centres unchanged path exercised
    assert any(p.n_confirm + p.n_reject > 0 for p in lib2.prototypes)
