# -*- coding: utf-8 -*-
"""Interpretable picking pattern library: induce → route → refine → update.

Loop
----
1. **Induce** precise patterns from dense run28 forwards (det/ρ/P-S/kernel knobs).
2. **Route** a new trace to the nearest prototype and apply a policy
   (skip / bypass NC / crop around coarse P / full fine).
3. **Refine** with dense HNF under that policy.
4. **Update** prototypes with high-confidence fine outcomes (online EMA).

This is intentionally lightweight and CPU-friendly for the library itself;
GPU is only needed when calling the picking model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# Kernel params (gamma/omega/wave_speed) are model-level constants — identical for
# every trace — so they carry no clustering signal and are deliberately excluded.
# rho and the energy ratio span many orders of magnitude, hence the log versions.
FEATURE_NAMES: tuple[str, ...] = (
    "det",
    "p_peak",
    "s_peak",
    "has_p",
    "has_s",
    "p_sec",
    "s_sec",
    "ps_gap_sec",
    "log_rho_mean",
    "log_rho_std",
    "log_rho_max",
    "energy_early",
    "energy_late",
    "log_energy_ratio",
)


@dataclass
class PatternPolicy:
    """What to do when this pattern fires."""

    name: str = "full_fine"
    skip_pick: bool = False
    bypass_noise_cancel: bool = False
    crop_around: str = "none"  # none | p | s | mid_ps
    crop_half_sec: float = 8.0
    # Cost scales with grid points, not physical duration: a shorter window on
    # fewer points is both cheaper and finer in dt than the full 60 s window.
    crop_len: Optional[int] = None
    local_window_sec: Optional[float] = None  # optional kernel override (unused in MVP crop path)


@dataclass
class PatternPrototype:
    pattern_id: int
    name: str
    center: list[float]
    count: int = 0
    n_event: int = 0
    n_noise: int = 0
    mean_ps_gap_sec: float = 0.0
    mean_p_mae_sec: float = 0.0
    mean_s_mae_sec: float = 0.0
    policy: PatternPolicy = field(default_factory=PatternPolicy)
    # Online update stats
    n_confirm: int = 0
    n_reject: int = 0


@dataclass
class RouteDecision:
    pattern_id: int
    name: str
    distance: float
    policy: PatternPolicy
    features: dict[str, float]


def _safe_peak_sec(probs: torch.Tensor, seq_len: int, window_sec: float, thr: float) -> tuple[float, float]:
    """Return (peak_prob, time_sec). time_sec=-1 if below threshold."""
    p = probs.detach().float().reshape(-1)
    peak = float(p.max().item())
    if peak < thr:
        return peak, -1.0
    idx = int(p.argmax().item())
    sec = float(idx) / max(seq_len - 1, 1) * window_sec
    return peak, sec


def waveform_energy_features(x: torch.Tensor, split: float = 0.35) -> tuple[float, float, float]:
    """x: (T, C) or (B, T, C) — use first batch row."""
    if x.dim() == 3:
        x = x[0]
    e = x.pow(2).mean(dim=-1)
    t = e.numel()
    cut = max(1, int(round(split * t)))
    early = float(e[:cut].mean().item())
    late = float(e[cut:].mean().item()) if cut < t else early
    ratio = early / max(late, 1e-8)
    return early, late, ratio


@torch.no_grad()
def extract_pattern_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    window_sec: float = 60.0,
    pick_threshold: float = 0.3,
    bypass_noise_cancel: bool = False,
) -> dict[str, float]:
    """One-trace feature vector from a dense (or NC-bypassed) forward.

    x: (1, T, C), t: (T, 1) or (1, T, 1)
    """
    was_bypass = getattr(model, "bypass_noise_cancel", False)
    if bypass_noise_cancel:
        model.bypass_noise_cancel = True
    try:
        if t.dim() == 2:
            t_in = t
        else:
            t_in = t[0]
        out = model(x, t_in)
    finally:
        model.bypass_noise_cancel = was_bypass

    seq_len = x.size(1)
    det = out["det"]
    det_p = torch.sigmoid(det)
    if det_p.dim() > 1:
        det_p = det_p.amax(dim=-1)
    det_v = float(det_p.reshape(-1)[0].item())

    p_prob = torch.sigmoid(out["p"][0])
    s_prob = torch.sigmoid(out["s"][0])
    p_peak, p_sec = _safe_peak_sec(p_prob, seq_len, window_sec, pick_threshold)
    s_peak, s_sec = _safe_peak_sec(s_prob, seq_len, window_sec, pick_threshold)
    if p_sec >= 0 and s_sec >= 0 and s_sec >= p_sec:
        gap = s_sec - p_sec
    else:
        gap = -1.0

    rho = out["rho"][0].detach().float()
    early, late, eratio = waveform_energy_features(x)

    gammas, omegas, cs = [], [], []
    if hasattr(model, "collect_kernel_params"):
        for _name, d in model.collect_kernel_params().items():
            gammas.append(float(d["gamma"]))
            omegas.append(float(d["omega"]))
            cs.append(float(d["wave_speed"]))
    g = float(np.mean(gammas)) if gammas else 0.5
    o = float(np.mean(omegas)) if omegas else 0.3
    c = float(np.mean(cs)) if cs else 6.0

    rho_mean = float(rho.mean().item())
    rho_std = float(rho.std(unbiased=False).item())
    rho_max = float(rho.max().item())

    feat = {
        "det": det_v,
        "p_peak": p_peak,
        "s_peak": s_peak,
        "has_p": 1.0 if p_sec >= 0 else 0.0,
        "has_s": 1.0 if s_sec >= 0 else 0.0,
        "p_sec": p_sec,
        "s_sec": s_sec,
        "ps_gap_sec": gap,
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "rho_max": rho_max,
        "log_rho_mean": _log10(rho_mean),
        "log_rho_std": _log10(rho_std),
        "log_rho_max": _log10(rho_max),
        "energy_early": early,
        "energy_late": late,
        "energy_ratio": eratio,
        "log_energy_ratio": _log10(eratio),
        "gamma_mean": g,
        "omega_mean": o,
        "c_mean": c,
    }
    return {k: (v if math.isfinite(v) else 0.0) for k, v in feat.items()}


def _log10(v: float) -> float:
    return float(np.log10(max(float(v), 1e-12)))


def features_to_vector(feat: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    vec = np.asarray([float(feat.get(n, 0.0)) for n in names], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _kmeans(x: np.ndarray, k: int, seed: int = 0, n_iter: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Simple numpy k-means with k-means++ init (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = x.shape[0]
    k = int(max(1, min(k, n)))
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(0, n)]
    for j in range(1, k):
        d2 = ((x[:, None, :] - centers[None, :j, :]) ** 2).sum(axis=-1).min(axis=1)
        total = d2.sum()
        if total <= 0:
            centers[j] = x[rng.integers(0, n)]
        else:
            centers[j] = x[rng.choice(n, p=d2 / total)]
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(n_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
            else:
                # Re-seed a dead cluster at the point furthest from its center.
                far = d2.min(axis=1).argmax()
                centers[j] = x[far]
    return labels, centers


def _zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def _policy_from_cluster_stats(
    *,
    n_event: int,
    n_noise: int,
    mean_gap: float,
    mean_det: float,
    mean_p_peak: float,
) -> PatternPolicy:
    n = max(n_event + n_noise, 1)
    noise_frac = n_noise / n
    if noise_frac >= 0.7 or (mean_det < 0.35 and mean_p_peak < 0.25):
        return PatternPolicy(
            name="noise_skip",
            skip_pick=True,
            bypass_noise_cancel=True,
            crop_around="none",
        )
    if mean_gap > 0 and mean_gap < 4.0 and mean_p_peak >= 0.35:
        return PatternPolicy(
            name="near_ps_crop_p",
            skip_pick=False,
            bypass_noise_cancel=False,
            crop_around="p",
            crop_half_sec=6.0,
            crop_len=400,
        )
    if mean_gap >= 8.0:
        return PatternPolicy(
            name="far_ps_crop_mid",
            skip_pick=False,
            bypass_noise_cancel=False,
            crop_around="mid_ps",
            crop_half_sec=12.0,
            crop_len=400,
        )
    if mean_det >= 0.8 and mean_p_peak < 0.35:
        # event-like det but weak P peak — keep full, bypass NC for speed probe
        return PatternPolicy(
            name="weak_p_full_bypass_nc",
            skip_pick=False,
            bypass_noise_cancel=True,
            crop_around="none",
        )
    return PatternPolicy(
        name="full_fine",
        skip_pick=False,
        bypass_noise_cancel=False,
        crop_around="none",
    )


class PatternLibrary:
    """Prototype bank + router + online EMA feedback."""

    def __init__(
        self,
        prototypes: list[PatternPrototype],
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        window_sec: float = 60.0,
        seq_len: int = 800,
        coarse_len: int = 200,
    ):
        self.prototypes = list(prototypes)
        self.feature_names = tuple(feature_names)
        d = len(self.feature_names)
        self.mean = np.zeros(d) if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = np.ones(d) if std is None else np.asarray(std, dtype=np.float64)
        self.window_sec = float(window_sec)
        self.seq_len = int(seq_len)
        self.coarse_len = int(coarse_len)

    @classmethod
    def build_from_feature_matrix(
        cls,
        feats: np.ndarray,
        *,
        is_event: Optional[np.ndarray] = None,
        p_mae: Optional[np.ndarray] = None,
        s_mae: Optional[np.ndarray] = None,
        k: int = 6,
        seed: int = 0,
        feature_names: Sequence[str] = FEATURE_NAMES,
        window_sec: float = 60.0,
        seq_len: int = 800,
        coarse_len: int = 200,
    ) -> "PatternLibrary":
        feats = np.asarray(feats, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[1] != len(feature_names):
            raise ValueError(f"feats shape {feats.shape} != (N, {len(feature_names)})")
        mu, sd = _zscore_fit(feats)
        z = (feats - mu) / sd
        labels, centers_z = _kmeans(z, k=k, seed=seed)
        centers = centers_z * sd + mu

        prototypes: list[PatternPrototype] = []
        for j in range(centers.shape[0]):
            mask = labels == j
            n = int(mask.sum())
            if n == 0:
                continue
            sub = feats[mask]
            n_event = int(is_event[mask].sum()) if is_event is not None else 0
            n_noise = n - n_event if is_event is not None else 0
            gaps = sub[:, feature_names.index("ps_gap_sec")]
            gaps = gaps[gaps >= 0]
            mean_gap = float(gaps.mean()) if gaps.size else -1.0
            mean_det = float(sub[:, feature_names.index("det")].mean())
            mean_p_peak = float(sub[:, feature_names.index("p_peak")].mean())
            mean_p_mae = float(p_mae[mask].mean()) if p_mae is not None else 0.0
            mean_s_mae = float(s_mae[mask].mean()) if s_mae is not None else 0.0
            policy = _policy_from_cluster_stats(
                n_event=n_event,
                n_noise=n_noise,
                mean_gap=mean_gap,
                mean_det=mean_det,
                mean_p_peak=mean_p_peak,
            )
            name = f"P{j}_{policy.name}"
            prototypes.append(
                PatternPrototype(
                    pattern_id=j,
                    name=name,
                    center=centers[j].tolist(),
                    count=n,
                    n_event=n_event,
                    n_noise=n_noise,
                    mean_ps_gap_sec=mean_gap,
                    mean_p_mae_sec=mean_p_mae,
                    mean_s_mae_sec=mean_s_mae,
                    policy=policy,
                )
            )
        return cls(
            prototypes,
            feature_names=feature_names,
            mean=mu,
            std=sd,
            window_sec=window_sec,
            seq_len=seq_len,
            coarse_len=coarse_len,
        )

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.mean) / self.std

    def route(self, feat: dict[str, float]) -> RouteDecision:
        if not self.prototypes:
            pol = PatternPolicy(name="full_fine")
            return RouteDecision(-1, "empty", 1e9, pol, feat)
        v = features_to_vector(feat, self.feature_names)
        z = self.normalize(v)
        best_i = 0
        best_d = 1e18
        for i, p in enumerate(self.prototypes):
            c = self.normalize(np.asarray(p.center, dtype=np.float64))
            d = float(np.linalg.norm(z - c))
            if d < best_d:
                best_d = d
                best_i = i
        proto = self.prototypes[best_i]
        return RouteDecision(proto.pattern_id, proto.name, best_d, proto.policy, feat)

    def update_from_fine(
        self,
        pattern_id: int,
        feat_fine: dict[str, float],
        *,
        confirmed: bool,
        ema: float = 0.05,
    ) -> None:
        """High-precision outcome feeds back into the matched prototype."""
        proto = None
        for p in self.prototypes:
            if p.pattern_id == pattern_id:
                proto = p
                break
        if proto is None:
            return
        if confirmed:
            proto.n_confirm += 1
            v = features_to_vector(feat_fine, self.feature_names)
            c = np.asarray(proto.center, dtype=np.float64)
            proto.center = ((1.0 - ema) * c + ema * v).tolist()
            proto.count += 1
        else:
            proto.n_reject += 1

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "window_sec": self.window_sec,
            "seq_len": self.seq_len,
            "coarse_len": self.coarse_len,
            "prototypes": [
                {
                    **{k: v for k, v in asdict(p).items() if k != "policy"},
                    "policy": asdict(p.policy),
                }
                for p in self.prototypes
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "PatternLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        protos = []
        for row in data["prototypes"]:
            pol = PatternPolicy(**row.pop("policy"))
            protos.append(PatternPrototype(policy=pol, **row))
        return cls(
            protos,
            feature_names=data.get("feature_names", FEATURE_NAMES),
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            window_sec=float(data.get("window_sec", 60.0)),
            seq_len=int(data.get("seq_len", 800)),
            coarse_len=int(data.get("coarse_len", 200)),
        )

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for p in self.prototypes:
            rows.append(
                {
                    "id": p.pattern_id,
                    "name": p.name,
                    "count": p.count,
                    "event": p.n_event,
                    "noise": p.n_noise,
                    "gap": round(p.mean_ps_gap_sec, 3),
                    "policy": p.policy.name,
                    "confirm": p.n_confirm,
                    "reject": p.n_reject,
                }
            )
        return rows


def downsample_trace(
    x: torch.Tensor,
    t: torch.Tensor,
    out_len: int,
    *,
    window_sec: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample a full trace onto a coarse grid for the cheap routing pass."""
    if t.dim() == 3:
        t = t[0]
    if x.size(1) == out_len:
        return x, t
    x_out = F.interpolate(
        x.transpose(1, 2), size=out_len, mode="linear", align_corners=False
    ).transpose(1, 2)
    t_out = torch.linspace(
        0.0, window_sec, out_len, device=x.device, dtype=x.dtype
    ).unsqueeze(-1)
    return x_out, t_out


def crop_trace_around_sec(
    x: torch.Tensor,
    t: torch.Tensor,
    center_sec: float,
    *,
    half_sec: float,
    window_sec: float,
    out_len: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Crop physical window around center_sec, then resample to out_len.

    Returns (x_out (1,out_len,C), t_out (out_len,1), shift_sec of crop start).
    """
    if x.dim() != 3 or x.size(0) != 1:
        raise ValueError("expected x shape (1, T, C)")
    seq = x.size(1)
    c = max(0.0, min(window_sec, float(center_sec)))
    lo = max(0.0, c - half_sec)
    hi = min(window_sec, c + half_sec)
    if hi - lo < 1e-3:
        lo, hi = 0.0, window_sec
    i0 = int(round(lo / window_sec * (seq - 1)))
    i1 = int(round(hi / window_sec * (seq - 1))) + 1
    i0 = max(0, min(seq - 1, i0))
    i1 = max(i0 + 2, min(seq, i1))
    crop = x[:, i0:i1, :]
    x_out = F.interpolate(
        crop.transpose(1, 2), size=out_len, mode="linear", align_corners=False
    ).transpose(1, 2)
    dur = hi - lo
    t_out = torch.linspace(0.0, dur, out_len, device=x.device, dtype=x.dtype).unsqueeze(-1)
    return x_out, t_out, lo


def apply_route_crop(
    x: torch.Tensor,
    t: torch.Tensor,
    feat: dict[str, float],
    policy: PatternPolicy,
    *,
    window_sec: float,
    out_len: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Maybe crop according to policy; always returns model-sized (out_len) tensors."""
    if policy.skip_pick or policy.crop_around == "none":
        if x.size(1) == out_len:
            return x, t if t.dim() == 2 else t[0], 0.0
        x_out = F.interpolate(
            x.transpose(1, 2), size=out_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        t_out = torch.linspace(0.0, window_sec, out_len, device=x.device, dtype=x.dtype).unsqueeze(-1)
        return x_out, t_out, 0.0

    if policy.crop_around == "p" and feat.get("p_sec", -1) >= 0:
        center = float(feat["p_sec"])
    elif policy.crop_around == "s" and feat.get("s_sec", -1) >= 0:
        center = float(feat["s_sec"])
    elif policy.crop_around == "mid_ps" and feat.get("p_sec", -1) >= 0 and feat.get("s_sec", -1) >= 0:
        center = 0.5 * (float(feat["p_sec"]) + float(feat["s_sec"]))
    else:
        center = window_sec * 0.5
    return crop_trace_around_sec(
        x,
        t if t.dim() == 2 else t[0],
        center,
        half_sec=policy.crop_half_sec,
        window_sec=window_sec,
        out_len=out_len,
    )
