# -*- coding: utf-8 -*-
"""Per-trace kernel-row summaries must vary with ρ, not with global γ/ω/c."""

from __future__ import annotations

import numpy as np
import torch

from hnf.kernel_response import (
    KERNEL_RESPONSE_NAMES,
    _summarize_weights,
    causal_incoming_row,
    extract_kernel_response_features,
)


class _FakeKern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_window_sec = 8.0

    def effective_gamma(self):
        return torch.tensor(0.4)

    def effective_omega(self):
        return torch.tensor(0.2)

    def dt_step_sec(self, n, t=None):
        return 60.0 / max(n - 1, 1)

    def window_bins(self, n, t=None):
        return min(n - 1, int(round(8.0 / self.dt_step_sec(n))))

    def _spherical_amplitude_mag(self, r):
        return (r.clamp_min(1e-6)).pow(-0.5)


def test_summarize_weights_uniform():
    lags = np.arange(1, 11, dtype=float)
    w = np.ones_like(lags)
    mean, spread, ent = _summarize_weights(w, lags)
    assert 4.0 < mean < 7.0
    assert spread > 0
    assert 0.5 < ent <= 1.0


def test_causal_row_depends_on_rho():
    kern = _FakeKern()
    n = 200
    t = torch.linspace(0, 60, n).view(n, 1)
    rho_flat = torch.ones(n) * 0.05
    rho_peak = rho_flat.clone()
    # strong attenuation in the immediate causal past of row 120
    rho_peak[100:120] = 5.0
    row = 120
    lags_a, w_a = causal_incoming_row(kern, n=n, t=t, rho=rho_flat, row_idx=row)
    lags_b, w_b = causal_incoming_row(kern, n=n, t=t, rho=rho_peak, row_idx=row)
    assert lags_a.size > 0 and lags_b.size > 0
    # L1 distance between normalised rows should be material
    pa = w_a / max(float(w_a.sum()), 1e-12)
    pb = w_b / max(float(w_b.sum()), 1e-12)
    assert float(np.abs(pa - pb).sum()) > 0.05


def test_kernel_response_names_stable():
    assert "p_kern_mean_lag_sec" in KERNEL_RESPONSE_NAMES
    assert len(KERNEL_RESPONSE_NAMES) == 7
