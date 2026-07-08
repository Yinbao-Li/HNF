# -*- coding: utf-8 -*-
"""On-the-fly synthetic dataset for Zhizi inversion bridge training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from hnf.inversion_1d import (
    LayeredEarth1D,
    default_station_distances,
    default_synth_model,
    synthesize_travel_times,
)
from hnf.inv_plot import perturb_initial
from hnf.synth_waveforms_1d import synthesize_multistation_batch


def random_layered_earth(
    device: torch.device | str,
    seed: int,
    base: LayeredEarth1D | None = None,
) -> LayeredEarth1D:
    """Random monotonic layered model around canonical synth."""
    base = base or default_synth_model(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vp, vs, q = perturb_initial(base.vp, base.vs, base.q, seed=seed, q_scale=1.0)
    scale = 0.95 + 0.1 * torch.rand((), generator=gen)
    vp = (vp * scale).clamp(min=1.5)
    for i in range(1, vp.numel()):
        vp[i] = torch.maximum(vp[i], vp[i - 1] + 0.05)
    vs = (vs * scale).clamp(min=1.0)
    for i in range(vs.numel()):
        vs[i] = torch.minimum(vs[i], vp[i] * 0.75)
    return LayeredEarth1D(depths=base.depths, vp=vp.to(device), vs=vs.to(device), q=q.to(device))


class ZhiziInversionDataset(Dataset):
    """
    Synthetic multi-station events for stage-1 inversion bridge training.

    Each item: waveforms (N, T, 3), travel times, true earth model.
    """

    def __init__(
        self,
        n_samples: int = 200,
        n_stations: int = 8,
        source_depth: float = 10.0,
        seq_len: int = 600,
        trace_noise: float = 0.05,
        time_noise: float = 0.02,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
        self.n_samples = n_samples
        self.n_stations = n_stations
        self.source_depth = source_depth
        self.seq_len = seq_len
        self.trace_noise = trace_noise
        self.time_noise = time_noise
        self.seed = seed
        self.device = torch.device(device)
        self.distances = default_station_distances(self.device, n_stations)
        self.depths = default_synth_model(self.device).depths

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float]:
        earth = random_layered_earth(self.device, seed=self.seed + idx * 9973)
        clean = synthesize_travel_times(earth, self.source_depth, self.distances)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + idx)
        obs_tp = clean["tp"] + self.time_noise * torch.randn(clean["tp"].shape, generator=gen, device=self.device)
        obs_ts = clean["ts"] + self.time_noise * torch.randn(clean["ts"].shape, generator=gen, device=self.device)
        x, t, _ = synthesize_multistation_batch(
            earth,
            self.source_depth,
            self.distances,
            seq_len=self.seq_len,
            noise_std=self.trace_noise,
            seed=self.seed + idx + 1000,
        )
        return {
            "x": x,
            "t": t,
            "obs_tp": obs_tp,
            "obs_ts": obs_ts,
            "true_vp": earth.vp,
            "true_vs": earth.vs,
            "true_q": earth.q,
            "depths": self.depths,
            "source_depth": self.source_depth,
            "distances": self.distances,
        }
