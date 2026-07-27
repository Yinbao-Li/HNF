# -*- coding: utf-8 -*-
"""Unit tests for EEG clinical helpers + taxonomy."""

from __future__ import annotations

import numpy as np

from hnf.eeg_clinical import benjamini_hochberg, welch_band_powers, rho_summaries
from hnf.eeg_dataset import (
    CLINICAL_LABEL_TO_ID,
    map_group_to_clinical,
    map_group_to_label,
)


def test_clinical_taxonomy_ftd_not_mci():
    assert map_group_to_clinical("F") == "FTD"
    assert map_group_to_clinical("A") == "AD"
    assert map_group_to_clinical("C") == "HC"
    assert map_group_to_label("F") == CLINICAL_LABEL_TO_ID["FTD"]
    assert map_group_to_label("F") == 1  # same slot as historical MCI


def test_bh_fdr_rejects_strong_signals():
    p = np.asarray([1e-6, 0.5, 0.8, 1e-5, 0.2], dtype=np.float64)
    rejected, q = benjamini_hochberg(p, alpha=0.05)
    assert rejected[0] and rejected[3]
    assert not rejected[1]


def test_band_power_and_rho_finite():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((19, 1280)).astype(np.float64)
    bp = welch_band_powers(x, 128.0)
    assert np.isfinite(bp["bp_alpha"])
    assert np.isfinite(bp["theta_alpha_ratio"])
    r = rho_summaries(np.abs(rng.standard_normal(1280)))
    assert np.isfinite(r["rho_mean"])
