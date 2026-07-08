# -*- coding: utf-8 -*-
"""Synthetic multi-station 3-component traces for picking / FWI experiments."""

from __future__ import annotations

import torch

from hnf.inversion_1d import LayeredEarth1D, travel_time_phase
from hnf.picking_metrics import idx_to_sec


def ricker_wavelet(t: torch.Tensor, f0: float, t0: float) -> torch.Tensor:
    """Ricker (Mexican hat) wavelet."""
    x = (torch.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * torch.exp(-x)


def synthesize_station_trace(
    tp_sec: float,
    ts_sec: float,
    seq_len: int = 800,
    window_sec: float = 60.0,
    f0_p: float = 2.5,
    f0_s: float = 1.5,
    amp_p: float = 1.0,
    amp_s: float = 0.7,
    noise_std: float = 0.05,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Build one 3-component trace (seq_len, 3) with P/S Ricker arrivals.

    Times outside [0, window_sec) are omitted.
    """
    dev = torch.device(device)
    t = torch.linspace(0.0, window_sec, seq_len, device=dev)
    trace = torch.zeros(seq_len, 3, device=dev)
    if 0.0 <= tp_sec < window_sec:
        wp = ricker_wavelet(t, f0_p, tp_sec)
        trace[:, 0] += amp_p * wp
        trace[:, 1] += 0.6 * amp_p * wp
        trace[:, 2] += 0.9 * amp_p * wp
    if 0.0 <= ts_sec < window_sec:
        ws = ricker_wavelet(t, f0_s, ts_sec)
        trace[:, 0] += 0.5 * amp_s * ws
        trace[:, 1] += amp_s * ws
        trace[:, 2] += 0.4 * amp_s * ws
    if noise_std > 0:
        gen = torch.Generator(device=dev)
        if seed is not None:
            gen.manual_seed(seed)
        trace = trace + noise_std * torch.randn(trace.shape, generator=gen, device=dev)
    # Per-trace normalization similar to STEAD preprocessing.
    std = trace.std().clamp(min=1e-4)
    trace = trace / std
    return trace


def synthesize_multistation_batch(
    model: LayeredEarth1D,
    source_depth: float,
    receiver_distances: torch.Tensor,
    seq_len: int = 800,
    window_sec: float = 60.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Returns
    -------
    x : (n_stations, seq_len, 3)
    t : (seq_len, 1) time coords for picking model
    meta : tp/ts in seconds and sample indices
    """
    dev = model.vp.device
    src = torch.tensor(source_depth, dtype=model.vp.dtype, device=dev)
    distances = receiver_distances.to(dev)
    tp = travel_time_phase(model, "P", src, distances)
    ts = travel_time_phase(model, "S", src, distances)
    traces = []
    gen_seed = seed
    for i in range(distances.numel()):
        traces.append(
            synthesize_station_trace(
                float(tp[i].item()),
                float(ts[i].item()),
                seq_len=seq_len,
                window_sec=window_sec,
                noise_std=noise_std,
                seed=gen_seed,
                device=dev,
            )
        )
        gen_seed += 1
    x = torch.stack(traces, dim=0)
    t = torch.linspace(0.0, window_sec, seq_len, device=dev).unsqueeze(-1)
    p_idx = (tp / window_sec * (seq_len - 1)).round().long().clamp(0, seq_len - 1)
    s_idx = (ts / window_sec * (seq_len - 1)).round().long().clamp(0, seq_len - 1)
    return x, t, {
        "tp_sec": tp,
        "ts_sec": ts,
        "p_idx": p_idx,
        "s_idx": s_idx,
        "tp_sec_list": [idx_to_sec(int(p_idx[i]), seq_len) for i in range(p_idx.numel())],
        "ts_sec_list": [idx_to_sec(int(s_idx[i]), seq_len) for i in range(s_idx.numel())],
    }
