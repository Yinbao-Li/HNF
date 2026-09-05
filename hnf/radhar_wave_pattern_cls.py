"""Wave v3 backbone + multi-cue spatial pattern head (late fusion).

Keeps the strong temporal Huygens classifier, and adds the diagnosed
jack/jump spatial-distribution detectors as a residual path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.radhar_huygens_cls import RadHARHuygensClsModel, build_time_axis
from hnf.radhar_io import ACTIVITIES
from hnf.radhar_spatial_wave_cls import (
    MultiCueEnergyFrontend,
    SpatialEnergyPatternHeadV2,
    default_range_centers_m,
    reshape_rich_to_range,
)


class WavePlusPatternClsModel(nn.Module):
    """``logits = wave_logits + softplus(α) * pattern_logits``."""

    def __init__(
        self,
        n_channels: int = 312,
        n_classes: int = len(ACTIVITIES),
        n_range_bins: int = 24,
        n_doppler_bins: int = 8,
        stem_ch: int = 64,
        embed_dim: int = 128,
        num_shared_layers: int = 4,
        epoch_sec: float = 4.0,
        local_window_sec: float = 0.5,
        dropout: float = 0.3,
        jack_jump_boost: bool = True,
        fuse_init: float = 0.3,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_range_bins = int(n_range_bins)
        self.n_doppler_bins = int(n_doppler_bins)
        self.epoch_sec = float(epoch_sec)

        self.wave = RadHARHuygensClsModel(
            n_channels=n_channels,
            n_classes=n_classes,
            stem_ch=stem_ch,
            embed_dim=embed_dim,
            num_shared_layers=num_shared_layers,
            epoch_sec=epoch_sec,
            local_window_sec=local_window_sec,
            dropout=dropout,
            medium_hidden=64,
            residual_energy=True,
            use_energy_pool=True,
            learnable_kernel_params=True,
        )
        self.register_buffer(
            "range_centers_m",
            default_range_centers_m(n_range_bins),
            persistent=False,
        )
        self.pattern_frontend = MultiCueEnergyFrontend(n_range_bins=n_range_bins)
        self.pattern_head = SpatialEnergyPatternHeadV2(
            n_range_bins=n_range_bins,
            n_classes=n_classes,
            dropout=dropout,
            jack_jump_boost=jack_jump_boost,
        )
        # α≈0 → behaves like pure Wave v3 at start
        self.fuse_logit = nn.Parameter(torch.tensor(float(fuse_init)).log())

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None, return_fields: bool = False
    ) -> dict[str, torch.Tensor]:
        if t is None:
            b, _, seq_len = x.shape
            t = build_time_axis(seq_len, self.epoch_sec, device=x.device)
            t = t.unsqueeze(0).expand(b, seq_len, 1)

        wave_out = self.wave(x, t, return_fields=return_fields)
        xr = reshape_rich_to_range(
            x, n_range_bins=self.n_range_bins, n_doppler_bins=self.n_doppler_bins
        )
        energies = self.pattern_frontend(xr)
        pattern_logits, diag = self.pattern_head(energies, self.range_centers_m)

        alpha = F.softplus(self.fuse_logit)
        logits = wave_out["logits"] + alpha * pattern_logits

        out = {
            "logits": logits,
            "wave_logits": wave_out["logits"],
            "pattern_logits": pattern_logits,
            "fuse_alpha": alpha.detach(),
            "width_amp": diag["width_amp"],
            "centroid_amp": diag["centroid_amp"],
            "jack_nearfar_amp": diag["jack_nearfar_amp"],
            "jump_nearfar_mean": diag["jump_nearfar_mean"],
            "jump_doppler_width_mean": diag["jump_doppler_width_mean"],
            "rho_mean": wave_out.get("rho_mean"),
        }
        if return_fields:
            out["energy_rt"] = energies["fused"]
            out["energies"] = energies
            if "energy_t" in wave_out:
                out["wave_energy_t"] = wave_out["energy_t"]
            if "jack_boost" in diag:
                out["jack_boost"] = diag["jack_boost"]
                out["jump_boost"] = diag["jump_boost"]
        return out
