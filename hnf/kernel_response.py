# -*- coding: utf-8 -*-
"""Per-trace Huygens kernel *responses* (not global γ/ω/c scalars).

Global kernel knobs are checkpoint-level constants — useless for clustering
traces. What varies per event is the **causal incoming weight row** at a pick
index, because ρ(t) modulates the sparse band. Summarising that row gives
interpretable, per-trace structure features:

- mean lag (effective support in seconds)
- lag spread
- normalised entropy of the weight distribution

Routing pattern libraries stay on cheap summary features; these features feed
the *interpretability* taxonomy (with causal-chain shape).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


KERNEL_RESPONSE_NAMES: tuple[str, ...] = (
    "p_kern_mean_lag_sec",
    "p_kern_spread_sec",
    "p_kern_entropy",
    "s_kern_mean_lag_sec",
    "s_kern_spread_sec",
    "s_kern_entropy",
    "ps_kern_lag_ratio",
)


def _summarize_weights(w: np.ndarray, lags: np.ndarray) -> tuple[float, float, float]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    lags = np.asarray(lags, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 1e-12 or lags.size == 0:
        return 0.0, 0.0, 0.0
    p = w / total
    mean = float((p * lags).sum())
    spread = float(np.sqrt(max(0.0, (p * (lags - mean) ** 2).sum())))
    ent = float(-(p * np.log(p + 1e-12)).sum())
    ent_norm = ent / max(np.log(float(lags.size)), 1e-12)
    return mean, spread, float(np.clip(ent_norm, 0.0, 1.0))


@torch.no_grad()
def causal_incoming_row(
    kernel: torch.nn.Module,
    *,
    n: int,
    t: torch.Tensor,
    rho: torch.Tensor,
    row_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute causal weights into ``row_idx`` over the sparse lag band.

    Returns ``(lags_sec, weights)`` as 1-d numpy arrays (empty if row has no past).
    """
    if row_idx <= 0 or n <= 1:
        return np.zeros(0), np.zeros(0)

    device = rho.device
    if t.dim() == 3:
        t = t[0]
    if t.dim() == 2 and t.size(-1) == 1:
        t_1d = t.reshape(-1)
    else:
        t_1d = t.reshape(-1)
    if t_1d.numel() != n:
        t_1d = torch.linspace(0.0, 60.0, n, device=device, dtype=torch.float32)

    if hasattr(kernel, "window_bins"):
        w_max = int(kernel.window_bins(n, t))
    else:
        dt = float((t_1d[-1] - t_1d[0]).item()) / max(n - 1, 1)
        local = float(getattr(kernel, "local_window_sec", 15.0) or 15.0)
        w_max = max(1, min(n - 1, int(round(local / max(dt, 1e-6)))))
    w_max = max(1, min(row_idx, w_max))

    if hasattr(kernel, "dt_step_sec"):
        dt = float(kernel.dt_step_sec(n, t))
    else:
        dt = float((t_1d[-1] - t_1d[0]).item()) / max(n - 1, 1)

    lags = torch.arange(1, w_max + 1, device=device, dtype=torch.float32) * dt
    src = row_idx - torch.arange(1, w_max + 1, device=device)
    valid = src >= 0
    if not bool(valid.any()):
        return np.zeros(0), np.zeros(0)
    src = src[valid]
    lags = lags[valid]

    if hasattr(kernel, "_spherical_amplitude_mag"):
        amp = kernel._spherical_amplitude_mag(lags.view(-1, 1, 1)).reshape(-1)
    else:
        amp = (lags + 1e-6).pow(-0.5)

    gamma = kernel.effective_gamma() if hasattr(kernel, "effective_gamma") else torch.tensor(0.5, device=device)
    omega = kernel.effective_omega() if hasattr(kernel, "effective_omega") else torch.tensor(0.3, device=device)
    amp = amp * torch.exp(-gamma * lags ** 2)
    amp = amp * (0.5 * (1.0 + torch.cos(omega * lags).abs()))

    rho_1d = rho.reshape(-1)
    if rho_1d.numel() == n:
        rho_i = rho_1d[row_idx]
        rho_j = rho_1d[src.long()]
        amp = amp * torch.exp(-(rho_i + rho_j) / 2.0 * lags)

    local = getattr(kernel, "local_window_sec", None)
    if local is not None:
        amp = amp * (lags <= float(local) + 1e-6).to(amp.dtype)

    return lags.detach().float().cpu().numpy(), amp.detach().float().cpu().numpy()


def _first_huygens_kernel(layers) -> Optional[torch.nn.Module]:
    if layers is None:
        return None
    for layer in layers:
        k = getattr(layer, "kernel", None)
        if k is not None and hasattr(k, "effective_gamma"):
            return k
    return None


@torch.no_grad()
def extract_kernel_response_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    p_idx: int,
    s_idx: int,
    bypass_noise_cancel: bool = True,
) -> dict[str, float]:
    """Per-trace kernel-row summaries at P and S indices.

    Uses ρ-modulated causal bands from the P/S branch kernels — **not** the
    global γ/ω/c scalars (those are identical for every trace).
    """
    out = {n: 0.0 for n in KERNEL_RESPONSE_NAMES}
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if t.dim() == 1:
        t = t.view(-1, 1)
    elif t.dim() == 3:
        t = t[0]

    was = getattr(model, "bypass_noise_cancel", False)
    model.bypass_noise_cancel = bool(bypass_noise_cancel)
    try:
        x_det, x_pick, _nc = model._apply_noise_cancel(x, t)
        rho = model.medium_net(x_pick)  # (B, T, 1)
    finally:
        model.bypass_noise_cancel = was

    n = int(x.size(1))
    rho0 = rho[0]

    p_kern = _first_huygens_kernel(getattr(model, "p_layers", None))
    s_kern = _first_huygens_kernel(getattr(model, "s_layers", None))
    if p_kern is None:
        p_kern = _first_huygens_kernel(getattr(model, "shared_layers", None))
    if s_kern is None:
        s_kern = p_kern

    def _one(kern, idx: int, prefix: str) -> None:
        if kern is None or idx < 0 or idx >= n:
            return
        lags, w = causal_incoming_row(kern, n=n, t=t, rho=rho0, row_idx=int(idx))
        mean, spread, ent = _summarize_weights(w, lags)
        out[f"{prefix}_kern_mean_lag_sec"] = mean
        out[f"{prefix}_kern_spread_sec"] = spread
        out[f"{prefix}_kern_entropy"] = ent

    _one(p_kern, p_idx, "p")
    _one(s_kern, s_idx, "s")
    p_m = out["p_kern_mean_lag_sec"]
    s_m = out["s_kern_mean_lag_sec"]
    out["ps_kern_lag_ratio"] = float(s_m / max(p_m, 1e-6)) if p_m > 0 else 0.0
    return out
