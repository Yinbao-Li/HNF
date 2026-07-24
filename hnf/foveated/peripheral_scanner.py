from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.kernel import HuygensKernel


@dataclass
class CandidateRegion:
    start: int
    end: int
    confidence: float


class PeripheralScanner(nn.Module):
    """Side-fovea global scanner over long windows.

    Input waveform: (B, 3, 6000)
    Output:
      - heatmap: (B, 6000) anomaly score in [0,1]
      - candidates: list[list[CandidateRegion]]
    """

    def __init__(
        self,
        seq_len: int = 6000,
        stride: int = 10,
        smooth_kernel: int = 31,
        detector: Literal["energy", "sparse_huygens"] = "energy",
        threshold_mode: Literal["topk", "fixed"] = "topk",
        topk_ratio: float = 0.20,
        fixed_threshold: float = 0.60,
        min_region: int = 30,
        merge_gap: int = 40,
        window_sec_full: float = 60.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.smooth_kernel = int(smooth_kernel)
        self.detector = detector
        self.threshold_mode = threshold_mode
        self.topk_ratio = float(topk_ratio)
        self.fixed_threshold = float(fixed_threshold)
        self.min_region = int(min_region)
        self.merge_gap = int(merge_gap)
        self.window_sec_full = float(window_sec_full)

        k = max(3, self.smooth_kernel | 1)
        self._smooth = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        with torch.no_grad():
            w = torch.ones(1, 1, k) / float(k)
            self._smooth.weight.copy_(w)
        for p in self._smooth.parameters():
            p.requires_grad_(False)

        # Sparse Huygens path: kernel applied on stride-subsampled nodes only.
        self.sparse_kernel = HuygensKernel(
            gamma=0.6,
            omega=0.7,
            wave_speed=6.0,
            distance_mode="time",
            local_window_sec=12.0,
            sparse_band=True,
            use_complex=True,
            principle="huygens",
        )
        self.sparse_proj = nn.Conv1d(3, 8, kernel_size=1)

    @staticmethod
    def _normalize01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def _energy_detector(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T)
        energy = torch.sqrt((x**2).mean(dim=1) + 1e-8)  # (B, T)
        # emphasize onsets
        d = F.pad(F.relu(energy[:, 1:] - energy[:, :-1]), (0, 1))
        score = 0.65 * energy + 0.35 * d
        return score

    def _sparse_huygens_detector(self, x: torch.Tensor) -> torch.Tensor:
        """Stride-subsampled Huygens scan, then upsample to full T."""
        b, _c, t = x.shape
        # (B, 3, T) -> subsample time
        xs = x[:, :, :: self.stride]  # (B, 3, Ts)
        ts_len = xs.size(-1)
        h = self.sparse_proj(xs).transpose(1, 2)  # (B, Ts, 8)
        dt = self.window_sec_full / max(self.seq_len - 1, 1)
        t_axis = (
            torch.arange(ts_len, device=x.device, dtype=x.dtype) * dt * self.stride
        ).view(1, ts_len, 1).expand(b, -1, -1)
        h_c = torch.complex(h, torch.zeros_like(h))
        y = self.sparse_kernel.forward_apply(h_c, x=h, t=t_axis)
        score_s = torch.abs(y).mean(dim=-1)  # (B, Ts)
        # Blend with energy onset prior on the same grid.
        energy_s = self._energy_detector(xs)
        score_s = 0.6 * score_s + 0.4 * energy_s
        score = score_s.repeat_interleave(self.stride, dim=-1)[..., :t]
        return score

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        return_candidates: bool = True,
    ) -> tuple[torch.Tensor, list[list[CandidateRegion]] | None]:
        if waveform.dim() != 3 or waveform.size(1) != 3:
            raise ValueError(f"Expected waveform (B,3,T), got {tuple(waveform.shape)}")
        _b, _c, t = waveform.shape
        if t != self.seq_len:
            raise ValueError(f"Expected T={self.seq_len}, got T={t}")

        if self.detector == "energy":
            raw = self._energy_detector(waveform)  # (B, T)
            if self.stride > 1:
                pooled = F.avg_pool1d(
                    raw.unsqueeze(1), kernel_size=self.stride, stride=self.stride
                ).squeeze(1)
                heat = pooled.repeat_interleave(self.stride, dim=-1)[..., :t]
            else:
                heat = raw
        elif self.detector == "sparse_huygens":
            heat = self._sparse_huygens_detector(waveform)
        else:
            raise ValueError(f"Unknown detector: {self.detector}")

        heat = self._smooth(heat.unsqueeze(1)).squeeze(1)
        heat = self._normalize01(heat)
        heat = heat.clamp(0.0, 1.0)

        candidates = self.extract_candidates(heat) if return_candidates else None
        return heat, candidates

    def extract_candidates(self, heatmap: torch.Tensor) -> list[list[CandidateRegion]]:
        if heatmap.dim() != 2 or heatmap.size(-1) != self.seq_len:
            raise ValueError(f"Expected heatmap (B,{self.seq_len}), got {tuple(heatmap.shape)}")
        b, t = heatmap.shape
        out: list[list[CandidateRegion]] = []
        for i in range(b):
            h = heatmap[i]
            if self.threshold_mode == "fixed":
                thr = torch.tensor(self.fixed_threshold, device=h.device, dtype=h.dtype)
            elif self.threshold_mode == "topk":
                q = 1.0 - max(1e-6, min(0.999, self.topk_ratio))
                thr = torch.quantile(h, q)
            else:
                raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

            mask = h >= thr
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() == 0:
                out.append([])
                continue

            starts = [int(idx[0].item())]
            ends: list[int] = []
            for j in range(1, idx.numel()):
                if int(idx[j].item()) > int(idx[j - 1].item()) + 1:
                    ends.append(int(idx[j - 1].item()) + 1)
                    starts.append(int(idx[j].item()))
            ends.append(int(idx[-1].item()) + 1)

            # merge close regions
            merged: list[tuple[int, int]] = []
            for s, e in zip(starts, ends):
                if not merged:
                    merged.append((s, e))
                    continue
                ps, pe = merged[-1]
                if s - pe <= self.merge_gap:
                    merged[-1] = (ps, e)
                else:
                    merged.append((s, e))

            regions: list[CandidateRegion] = []
            for s, e in merged:
                if e - s < self.min_region:
                    continue
                conf = float(h[s:e].mean().detach().cpu())
                regions.append(CandidateRegion(start=s, end=e, confidence=conf))
            regions.sort(key=lambda r: r.confidence, reverse=True)
            out.append(regions)
        return out

