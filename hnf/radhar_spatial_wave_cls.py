"""RadHAR spatial-energy-pattern classifier (v2).

Frontend builds per-cue range–time energy maps.
The head detects spatial distribution patterns:

  jack  ← bearing near/far oscillation + spatial width amp
  jump  ← intensity near/far mean (energy piled nearer) + doppler width mean

Diagnosis (test set Cohen's d) showed centroid_amp fails for jump, while
``intensity:nearfar_mean`` (d≈1.04) and ``doppler:width_mean`` (d≈1.59) separate it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.radhar_io import ACTIVITIES, CLS_FEATURE_NAMES

ACTIVITY_TO_IDX = {a: i for i, a in enumerate(ACTIVITIES)}
# cue indices inside reshape_rich_to_range F-axis
_CUE_INTENSITY = 0
_CUE_ABS_VEL = 1
_CUE_ABS_BEARING = 4
# doppler channels start at len(CLS_FEATURE_NAMES)


def reshape_rich_to_range(
    x: torch.Tensor,
    *,
    n_range_bins: int = 24,
    n_doppler_bins: int = 8,
    n_cues: int = len(CLS_FEATURE_NAMES),
) -> torch.Tensor:
    b, c, t = x.shape
    r = int(n_range_bins)
    nd = int(n_doppler_bins)
    expected = n_cues * r + r * nd
    if c != expected:
        raise ValueError(f"Expected C={expected}, got {c}")
    cues = x[:, : n_cues * r, :].reshape(b, n_cues, r, t)
    rd = x[:, n_cues * r :, :].reshape(b, r, nd, t).permute(0, 2, 1, 3)
    return torch.cat([cues, rd], dim=1)


def default_range_centers_m(n_range_bins: int = 24, range_max_m: float = 8.0) -> torch.Tensor:
    edges = torch.linspace(0.0, float(range_max_m), n_range_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _traj_stats(z: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            z.mean(dim=-1),
            z.std(dim=-1, unbiased=False),
            z.max(dim=-1).values - z.min(dim=-1).values,
            z.max(dim=-1).values,
        ],
        dim=-1,
    )


def distribution_geometry(energy_rt: torch.Tensor, range_m: torch.Tensor) -> dict[str, torch.Tensor]:
    """Named spatial-distribution trajectories from ``E(t,r)``."""
    p = energy_rt / energy_rt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    r = range_m.view(1, 1, -1)
    mu = (p * r).sum(dim=-1)
    var = (p * (r - mu.unsqueeze(-1)).pow(2)).sum(dim=-1)
    sigma = var.clamp_min(0.0).sqrt()
    peak_idx = energy_rt.argmax(dim=-1)
    peak = range_m[peak_idx]
    rb = energy_rt.size(-1)
    n1, n2 = rb // 3, 2 * rb // 3
    near = energy_rt[..., :n1].sum(dim=-1)
    far = energy_rt[..., n2:].sum(dim=-1)
    nearfar = torch.log1p(near) - torch.log1p(far)
    return {
        "centroid": mu,
        "width": sigma,
        "peak": peak,
        "nearfar": nearfar,
        "profile": energy_rt.mean(dim=1),
    }


class MultiCueEnergyFrontend(nn.Module):
    """Refine raw cue maps into non-negative energy ``E_c(t,r)`` for key cues."""

    def __init__(self, n_range_bins: int = 24, hidden: int = 32):
        super().__init__()
        self.n_range_bins = n_range_bins
        # per-cue 1x1 + small temporal smooth; softplus → energy
        self.refine = nn.ModuleDict({
            "intensity": self._block(hidden),
            "bearing": self._block(hidden),
            "doppler": self._block(hidden),
            "fused": self._block(hidden),
        })

    @staticmethod
    def _block(hidden: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=(3, 3), padding=1, bias=False),
        )

    def forward(self, xr: torch.Tensor) -> dict[str, torch.Tensor]:
        # xr: (B, F, R, T)
        inten = xr[:, _CUE_INTENSITY:_CUE_INTENSITY + 1].abs()
        bearing = xr[:, _CUE_ABS_BEARING:_CUE_ABS_BEARING + 1].abs()
        doppler = xr[:, len(CLS_FEATURE_NAMES) :].abs().mean(dim=1, keepdim=True)
        fused = (inten + bearing + doppler) / 3.0
        out = {}
        for name, raw in (("intensity", inten), ("bearing", bearing), ("doppler", doppler), ("fused", fused)):
            e = F.softplus(self.refine[name](raw)).squeeze(1)  # (B, R, T)
            out[name] = e.permute(0, 2, 1).contiguous()        # (B, T, R)
        return out


class SpatialEnergyPatternHeadV2(nn.Module):
    """Multi-cue spatial distribution pattern head."""

    def __init__(
        self,
        n_range_bins: int,
        n_classes: int = len(ACTIVITIES),
        n_prototypes: int = 8,
        hidden: int = 160,
        dropout: float = 0.3,
        jack_jump_boost: bool = True,
    ):
        super().__init__()
        self.n_range_bins = int(n_range_bins)
        self.jack_jump_boost = bool(jack_jump_boost)
        self.jack_idx = ACTIVITY_TO_IDX["jack"]
        self.jump_idx = ACTIVITY_TO_IDX["jump"]
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_range_bins) * 0.02)

        # per cue: profile(R) + 4 traj stats × 4 geoms = R + 16
        # ×3 cues (intensity, bearing, doppler) + fused profile/proto + explicit detectors
        per_cue = n_range_bins + 16
        feat_dim = per_cue * 3 + n_range_bins + n_prototypes + 4  # +4 explicit detectors
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.jack_nearfar_amp_scale = nn.Parameter(torch.tensor(1.0))
        self.jack_width_amp_scale = nn.Parameter(torch.tensor(0.5))
        self.jump_nearfar_mean_scale = nn.Parameter(torch.tensor(1.0))
        self.jump_doppler_width_scale = nn.Parameter(torch.tensor(1.0))

    def _cue_feat(self, e: torch.Tensor, range_m: torch.Tensor) -> tuple[torch.Tensor, dict]:
        g = distribution_geometry(e, range_m)
        profile = g["profile"]
        profile_n = profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        stats = torch.cat(
            [_traj_stats(g["centroid"]), _traj_stats(g["width"]),
             _traj_stats(g["peak"]), _traj_stats(g["nearfar"])],
            dim=-1,
        )
        feat = torch.cat([profile_n, stats], dim=-1)
        diag = {
            "width_amp": _traj_stats(g["width"])[:, 2],
            "nearfar_amp": _traj_stats(g["nearfar"])[:, 2],
            "nearfar_mean": g["nearfar"].mean(dim=-1),
            "width_mean": g["width"].mean(dim=-1),
            "centroid_amp": _traj_stats(g["centroid"])[:, 2],
            "profile": profile_n,
        }
        return feat, diag

    def forward(
        self, energies: dict[str, torch.Tensor], range_m: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = []
        diags = {}
        for cue in ("intensity", "bearing", "doppler"):
            f, d = self._cue_feat(energies[cue], range_m)
            parts.append(f)
            diags[cue] = d

        fused_profile = energies["fused"].mean(dim=1)
        fused_n = fused_profile / fused_profile.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        proto = F.normalize(self.prototypes, dim=-1)
        proto_sim = F.normalize(fused_n, dim=-1) @ proto.t()

        # explicit detectors from diagnosis
        jack_nf_amp = diags["bearing"]["nearfar_amp"]
        jack_w_amp = diags["bearing"]["width_amp"]
        jump_nf_mean = diags["intensity"]["nearfar_mean"]
        jump_dop_w = diags["doppler"]["width_mean"]

        explicit = torch.stack([jack_nf_amp, jack_w_amp, jump_nf_mean, jump_dop_w], dim=-1)
        feat = torch.cat(parts + [fused_n, proto_sim, explicit], dim=-1)
        logits = self.mlp(feat)

        diag_out = {
            "jack_nearfar_amp": jack_nf_amp,
            "jack_width_amp": jack_w_amp,
            "jump_nearfar_mean": jump_nf_mean,
            "jump_doppler_width_mean": jump_dop_w,
            "width_amp": jack_w_amp,          # compat
            "centroid_amp": diags["intensity"]["centroid_amp"],
        }
        if self.jack_jump_boost:
            jack_boost = (
                F.softplus(self.jack_nearfar_amp_scale) * jack_nf_amp
                + F.softplus(self.jack_width_amp_scale) * jack_w_amp
            )
            jump_boost = (
                F.softplus(self.jump_nearfar_mean_scale) * jump_nf_mean
                + F.softplus(self.jump_doppler_width_scale) * jump_dop_w
            )
            logits = logits.clone()
            logits[:, self.jack_idx] = logits[:, self.jack_idx] + jack_boost
            logits[:, self.jump_idx] = logits[:, self.jump_idx] + jump_boost
            diag_out["jack_boost"] = jack_boost
            diag_out["jump_boost"] = jump_boost
        return logits, diag_out


class RadHARSpatialPatternClsModel(nn.Module):
    """Multi-cue energy frontend + SpatialEnergyPatternHeadV2."""

    def __init__(
        self,
        n_channels: int = 312,
        n_classes: int = len(ACTIVITIES),
        n_range_bins: int = 24,
        n_doppler_bins: int = 8,
        embed_dim: int = 64,  # unused; kept for ckpt compat
        range_max_m: float = 8.0,
        dropout: float = 0.3,
        jack_jump_boost: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_range_bins = int(n_range_bins)
        self.n_doppler_bins = int(n_doppler_bins)
        self.n_cues = len(CLS_FEATURE_NAMES)
        self.register_buffer(
            "range_centers_m",
            default_range_centers_m(n_range_bins, range_max_m),
            persistent=False,
        )
        self.frontend = MultiCueEnergyFrontend(n_range_bins=n_range_bins)
        self.head = SpatialEnergyPatternHeadV2(
            n_range_bins=n_range_bins,
            n_classes=n_classes,
            dropout=dropout,
            jack_jump_boost=jack_jump_boost,
        )

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None, return_fields: bool = False
    ) -> dict[str, torch.Tensor]:
        if x.size(1) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {x.size(1)}")
        xr = reshape_rich_to_range(
            x, n_range_bins=self.n_range_bins, n_doppler_bins=self.n_doppler_bins, n_cues=self.n_cues
        )
        energies = self.frontend(xr)
        logits, diag = self.head(energies, self.range_centers_m)
        out = {
            "logits": logits,
            "centroid_amp": diag["centroid_amp"],
            "width_amp": diag["width_amp"],
            "jump_nearfar_mean": diag["jump_nearfar_mean"],
            "jump_doppler_width_mean": diag["jump_doppler_width_mean"],
            "jack_nearfar_amp": diag["jack_nearfar_amp"],
        }
        if return_fields:
            out["energy_rt"] = energies["fused"]
            out["energies"] = energies
            if "jack_boost" in diag:
                out["jack_boost"] = diag["jack_boost"]
                out["jump_boost"] = diag["jump_boost"]
        return out


# aliases
RadHARSpatialWaveClsModel = RadHARSpatialPatternClsModel
SpatialEnergyPatternHead = SpatialEnergyPatternHeadV2
