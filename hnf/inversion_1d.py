# -*- coding: utf-8 -*-
"""1D layered Earth forward modeling and vp/vs/Q inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


Phase = Literal["P", "S"]


@dataclass
class LayeredEarth1D:
    """Piecewise-constant 1D model with interfaces at fixed depths (km)."""

    depths: torch.Tensor  # (n_layers + 1,) increasing, depths[0]=0
    vp: torch.Tensor  # (n_layers,) km/s
    vs: torch.Tensor  # (n_layers,) km/s
    q: torch.Tensor  # (n_layers,) quality factor

    def __post_init__(self) -> None:
        n = self.depths.numel() - 1
        if self.vp.shape != (n,) or self.vs.shape != (n,) or self.q.shape != (n,):
            raise ValueError("vp/vs/q must have shape (n_layers,)")

    @property
    def n_layers(self) -> int:
        return self.depths.numel() - 1

    @property
    def thicknesses(self) -> torch.Tensor:
        return self.depths[1:] - self.depths[:-1]

    def velocity(self, phase: Phase) -> torch.Tensor:
        return self.vp if phase == "P" else self.vs


def layer_index_at_depth(depths: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return layer index for each depth z (0-based)."""
    idx = torch.searchsorted(depths, z, right=True) - 1
    return idx.clamp(min=0, max=depths.numel() - 2)


def vertical_integral_slowness(
    depths: torch.Tensor,
    velocity: torch.Tensor,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
) -> torch.Tensor:
    """
    Integrate dz / v(z) between two depths along a vertical ray.
    depths: (L+1,), velocity: (L,), z_start/z_end: scalar or (B,)
    """
    if z_start.dim() == 0:
        z_start = z_start.unsqueeze(0)
    if z_end.dim() == 0:
        z_end = z_end.unsqueeze(0)
    lo = torch.minimum(z_start, z_end)
    hi = torch.maximum(z_start, z_end)
    thick = depths[1:] - depths[:-1]
    z_lo = depths[:-1]
    z_hi = depths[1:]
    overlap = (torch.minimum(hi.unsqueeze(-1), z_hi) - torch.maximum(lo.unsqueeze(-1), z_lo)).clamp(min=0.0)
    slowness = 1.0 / velocity
    return (overlap * slowness).sum(dim=-1)


def ray_horizontal_distance(
    depths: torch.Tensor,
    velocity: torch.Tensor,
    source_depth: torch.Tensor,
    ray_param: torch.Tensor,
) -> torch.Tensor:
    """
    Horizontal distance for a ray leaving the source upward to the surface.
    ray_param p = sin(theta)/v is constant (Snell's law).
    """
    v = velocity
    p = ray_param.unsqueeze(-1)
    pv = p * v
    cos_i = torch.sqrt((1.0 - (pv * pv)).clamp(min=1e-8))
    dz = _upward_layer_segments(depths, source_depth)
    return (dz * pv * v / cos_i).sum(dim=-1)


def ray_travel_time_upward(
    depths: torch.Tensor,
    velocity: torch.Tensor,
    source_depth: torch.Tensor,
    ray_param: torch.Tensor,
) -> torch.Tensor:
    """Travel time from source depth to surface along a ray with parameter p."""
    v = velocity
    p = ray_param.unsqueeze(-1)
    pv = p * v
    cos_i = torch.sqrt((1.0 - (pv * pv)).clamp(min=1e-8))
    dz = _upward_layer_segments(depths, source_depth)
    return (dz / (v * cos_i)).sum(dim=-1)


def _upward_layer_segments(depths: torch.Tensor, source_depth: torch.Tensor) -> torch.Tensor:
    """Vertical segment length (km) traversed in each layer on an upward ray."""
    z_top = depths[:-1]
    z_bot = depths[1:]
    src = source_depth.unsqueeze(-1)
    return (torch.minimum(z_bot, src) - z_top).clamp(min=0.0)


def find_ray_parameter(
    depths: torch.Tensor,
    velocity: torch.Tensor,
    source_depth: torch.Tensor,
    target_distance: torch.Tensor,
    n_iter: int = 40,
) -> torch.Tensor:
    """
    Solve for ray parameter p so upward ray reaches horizontal distance target_distance.
    Uses differentiable bisection on p in [0, p_max).
    """
    if target_distance.dim() == 0:
        target_distance = target_distance.unsqueeze(0)
    v_min = velocity.min()
    p_max = 0.98 / v_min
    lo = torch.zeros_like(target_distance)
    hi = torch.ones_like(target_distance) * p_max
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        dx = ray_horizontal_distance(depths, velocity, source_depth, mid)
        lo = torch.where(dx < target_distance, mid, lo)
        hi = torch.where(dx >= target_distance, mid, hi)
    return 0.5 * (lo + hi)


def intrinsic_attenuation_log(
    depths: torch.Tensor,
    q: torch.Tensor,
    velocity: torch.Tensor,
    source_depth: torch.Tensor,
    ray_param: torch.Tensor,
    frequency_hz: float,
) -> torch.Tensor:
    """Log spectral amplitude: -pi * f * sum(dt_i / Q_i) along the upward ray."""
    v = velocity
    p = ray_param.unsqueeze(-1)
    pv = p * v
    cos_i = torch.sqrt((1.0 - (pv * pv)).clamp(min=1e-8))
    dz = _upward_layer_segments(depths, source_depth)
    dt_layer = dz / (v * cos_i)
    return -torch.pi * frequency_hz * (dt_layer / q.clamp(min=1.0)).sum(dim=-1)


def _ray_params(
    depths: torch.Tensor,
    velocity: torch.Tensor,
    source_depth: torch.Tensor,
    receiver_distance: torch.Tensor,
) -> torch.Tensor:
    if receiver_distance.dim() == 0:
        receiver_distance = receiver_distance.unsqueeze(0)
    src = source_depth if source_depth.dim() else source_depth.unsqueeze(0)
    if src.numel() == 1 and receiver_distance.numel() > 1:
        src = src.expand_as(receiver_distance)
    p = torch.zeros_like(receiver_distance)
    nonzero = receiver_distance.abs() >= 1e-6
    if nonzero.any():
        p[nonzero] = find_ray_parameter(
            depths, velocity, src[nonzero], receiver_distance[nonzero]
        )
    return p, src


def amplitude_log_phase(
    model: LayeredEarth1D,
    phase: Phase,
    source_depth: torch.Tensor,
    receiver_distance: torch.Tensor,
    frequency_hz: float | torch.Tensor,
) -> torch.Tensor:
    """Log amplitude for direct upward ray (frequency-domain attenuation)."""
    v = model.velocity(phase)
    p, src = _ray_params(model.depths, v, source_depth, receiver_distance)
    if isinstance(frequency_hz, (float, int)):
        return intrinsic_attenuation_log(
            model.depths, model.q, v, src, p, float(frequency_hz)
        )
    freqs = frequency_hz
    logs = []
    for f in freqs:
        logs.append(
            intrinsic_attenuation_log(model.depths, model.q, v, src, p, float(f))
        )
    return torch.stack(logs, dim=-1)


def travel_time_phase(
    model: LayeredEarth1D,
    phase: Phase,
    source_depth: torch.Tensor,
    receiver_distance: torch.Tensor,
) -> torch.Tensor:
    """
    First-arrival travel time for direct upward ray to surface at horizontal offset.
    Zero offset uses vertical integration; nonzero uses bent-ray tracing.
    """
    v = model.velocity(phase)
    if receiver_distance.dim() == 0:
        receiver_distance = receiver_distance.unsqueeze(0)
    src = source_depth if source_depth.dim() else source_depth.unsqueeze(0)
    if src.numel() == 1 and receiver_distance.numel() > 1:
        src = src.expand_as(receiver_distance)

    times = torch.empty_like(receiver_distance, dtype=v.dtype, device=v.device)
    zero_mask = receiver_distance.abs() < 1e-6
    if zero_mask.any():
        times[zero_mask] = vertical_integral_slowness(
            model.depths, v, src[zero_mask], torch.zeros_like(src[zero_mask])
        )
    if (~zero_mask).any():
        r = receiver_distance[~zero_mask]
        s = src[~zero_mask]
        p = find_ray_parameter(model.depths, v, s, r)
        times[~zero_mask] = ray_travel_time_upward(model.depths, v, s, p)
    return times


def synthesize_observations(
    model: LayeredEarth1D,
    source_depth: float,
    receiver_distances: torch.Tensor,
    frequency_hz: float | list[float] = 8.0,
    time_noise_std: float = 0.0,
    amp_noise_std: float = 0.0,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate P/S travel times and log-amplitudes at sparse receivers."""
    freqs = (
        torch.tensor(frequency_hz, dtype=model.vp.dtype, device=model.vp.device)
        if isinstance(frequency_hz, (list, tuple))
        else frequency_hz
    )
    src = torch.tensor(source_depth, dtype=model.vp.dtype, device=model.vp.device)
    tp = travel_time_phase(model, "P", src, receiver_distances)
    ts = travel_time_phase(model, "S", src, receiver_distances)
    log_ap = amplitude_log_phase(model, "P", src, receiver_distances, freqs)
    log_as = amplitude_log_phase(model, "S", src, receiver_distances, freqs)
    if time_noise_std > 0 or amp_noise_std > 0:
        gen = torch.Generator(device=model.vp.device)
        if seed is not None:
            gen.manual_seed(seed)
        if time_noise_std > 0:
            tp = tp + time_noise_std * torch.randn(tp.shape, generator=gen, device=tp.device, dtype=tp.dtype)
            ts = ts + time_noise_std * torch.randn(ts.shape, generator=gen, device=ts.device, dtype=ts.dtype)
        if amp_noise_std > 0:
            log_ap = log_ap + amp_noise_std * torch.randn(log_ap.shape, generator=gen, device=log_ap.device, dtype=log_ap.dtype)
            log_as = log_as + amp_noise_std * torch.randn(log_as.shape, generator=gen, device=log_as.device, dtype=log_as.dtype)
    return {"tp": tp, "ts": ts, "log_ap": log_ap, "log_as": log_as, "frequencies_hz": freqs}


def synthesize_travel_times(
    model: LayeredEarth1D,
    source_depth: float,
    receiver_distances: torch.Tensor,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Backward-compatible travel-time-only synthesis."""
    obs = synthesize_observations(
        model, source_depth, receiver_distances,
        time_noise_std=noise_std, seed=seed,
    )
    return {"tp": obs["tp"], "ts": obs["ts"]}


class InvertibleLayeredEarth1D(nn.Module):
    """
    Learnable 1D layered model.

    vp and Q increase with depth via positive layer increments.
    vs = vp * bounded per-layer P/S velocity ratios.
    """

    def __init__(
        self,
        depths: torch.Tensor,
        vp_init: torch.Tensor,
        vs_init: torch.Tensor,
        q_init: torch.Tensor,
        invert_q: bool = False,
    ):
        super().__init__()
        self.invert_q = invert_q
        self.register_buffer("depths", depths)
        n = depths.numel() - 1
        vp_inc = torch.clamp(vp_init[1:] - vp_init[:-1], min=0.05)
        self.log_vp_base = nn.Parameter(torch.log(vp_init[0].clamp(min=1.0)))
        self.log_vp_inc = nn.Parameter(torch.log(vp_inc))
        ratio = (vs_init / vp_init.clamp(min=1e-6)).clamp(0.35, 0.75)
        self.logit_vs_ratio = nn.Parameter(torch.log(ratio / (1.0 - ratio)))
        if invert_q:
            q_inc = torch.clamp(q_init[1:] - q_init[:-1], min=5.0)
            self.log_q_base = nn.Parameter(torch.log(q_init[0].clamp(min=20.0)))
            self.log_q_inc = nn.Parameter(torch.log(q_inc))
        else:
            self.register_buffer("q_fixed", q_init)

    @property
    def vp(self) -> torch.Tensor:
        base = torch.exp(self.log_vp_base)
        inc = torch.exp(self.log_vp_inc)
        return base + torch.cat([torch.zeros(1, device=base.device), torch.cumsum(inc, dim=0)])

    @property
    def vs(self) -> torch.Tensor:
        ratio = torch.sigmoid(self.logit_vs_ratio)
        return self.vp * ratio

    @property
    def q(self) -> torch.Tensor:
        if self.invert_q:
            base = torch.exp(self.log_q_base)
            inc = torch.exp(self.log_q_inc)
            return base + torch.cat([torch.zeros(1, device=base.device), torch.cumsum(inc, dim=0)])
        return self.q_fixed

    @property
    def earth(self) -> LayeredEarth1D:
        return LayeredEarth1D(
            depths=self.depths,
            vp=self.vp,
            vs=self.vs,
            q=self.q,
        )

    def forward(
        self,
        source_depth: torch.Tensor,
        receiver_distances: torch.Tensor,
        frequency_hz: float | list[float] = 8.0,
        with_amplitude: bool = False,
    ) -> dict[str, torch.Tensor]:
        model = self.earth
        out = {
            "tp": travel_time_phase(model, "P", source_depth, receiver_distances),
            "ts": travel_time_phase(model, "S", source_depth, receiver_distances),
        }
        if with_amplitude:
            out["log_ap"] = amplitude_log_phase(
                model, "P", source_depth, receiver_distances, frequency_hz
            )
            out["log_as"] = amplitude_log_phase(
                model, "S", source_depth, receiver_distances, frequency_hz
            )
        return out


def inversion_loss(
    pred: dict[str, torch.Tensor],
    obs: dict[str, torch.Tensor],
    model: InvertibleLayeredEarth1D,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor | None = None,
    smooth_weight: float = 0.05,
    anchor_weight: float = 0.005,
    amp_weight: float = 0.0,
    q_anchor_weight: float = 0.002,
    hnf_reg: torch.Tensor | None = None,
    hnf_weight: float = 0.0,
    times_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Travel-time (+ optional amplitude / HNF) misfit with regularization."""
    loss_tp = torch.mean((pred["tp"] - obs["tp"]) ** 2)
    loss_ts = torch.mean((pred["ts"] - obs["ts"]) ** 2)
    earth = model.earth
    smooth = torch.mean(torch.exp(model.log_vp_inc) ** 2)
    if model.invert_q:
        smooth = smooth + 0.01 * torch.mean(torch.exp(model.log_q_inc) ** 2)
    anchor = torch.mean((earth.vp - vp_init) ** 2) + torch.mean((earth.vs - vs_init) ** 2)
    q_anchor = torch.tensor(0.0, device=earth.vp.device)
    if q_init is not None and model.invert_q:
        q_anchor = torch.mean((earth.q - q_init) ** 2)
    if times_weight <= 0.0:
        total = amp_weight * (loss_tp * 0.0)
        if amp_weight > 0 and "log_ap" in obs:
            loss_amp_p = torch.mean((pred["log_ap"] - obs["log_ap"]) ** 2)
            loss_amp_s = torch.mean((pred["log_as"] - obs["log_as"]) ** 2)
            total = amp_weight * (loss_amp_p + loss_amp_s)
            metrics = {
                "loss_tp": float(loss_tp.detach()),
                "loss_ts": float(loss_ts.detach()),
                "loss_amp_p": float(loss_amp_p.detach()),
                "loss_amp_s": float(loss_amp_s.detach()),
            }
        else:
            metrics = {"loss_tp": float(loss_tp.detach()), "loss_ts": float(loss_ts.detach())}
        total = total + q_anchor_weight * q_anchor
        metrics.update({
            "loss": float(total.detach()),
            "smooth": 0.0,
            "anchor": float(q_anchor.detach()),
        })
        return total, metrics
    total = times_weight * (loss_tp + loss_ts) + smooth_weight * smooth + anchor_weight * anchor
    if q_init is not None and model.invert_q:
        total = total + q_anchor_weight * q_anchor
    metrics = {
        "loss": float(total.detach()),
        "loss_tp": float(loss_tp.detach()),
        "loss_ts": float(loss_ts.detach()),
        "smooth": float(smooth.detach()),
        "anchor": float(anchor.detach()),
    }
    if amp_weight > 0 and "log_ap" in obs:
        loss_amp_p = torch.mean((pred["log_ap"] - obs["log_ap"]) ** 2)
        loss_amp_s = torch.mean((pred["log_as"] - obs["log_as"]) ** 2)
        total = total + amp_weight * (loss_amp_p + loss_amp_s)
        metrics["loss_amp_p"] = float(loss_amp_p.detach())
        metrics["loss_amp_s"] = float(loss_amp_s.detach())
    if hnf_reg is not None and hnf_weight > 0:
        total = total + hnf_weight * hnf_reg
        metrics["loss_hnf"] = float(hnf_reg.detach())
    metrics["loss"] = float(total.detach())
    return total, metrics


def layer_center_depths(depths: torch.Tensor) -> torch.Tensor:
    return 0.5 * (depths[:-1] + depths[1:])


def depth_coords_1d(depths_km: torch.Tensor) -> torch.Tensor:
    """Map depth (km) to 2D coords for HuygensNeuralField: [depth_norm, 0]."""
    dmax = depths_km.max().clamp(min=1e-6)
    d = depths_km / dmax
    return torch.stack([d, torch.zeros_like(d)], dim=-1)


def hnf_profile_regularization(
    model: InvertibleLayeredEarth1D,
    hnf_field,
    n_grid: int = 64,
) -> torch.Tensor:
    """
    HNF kernel smoothness prior on vp/vs profiles.

    Combines grid-gradient smoothness with tying layer values to
    Huygens-interpolated samples at layer centers.
    """
    depths = model.depths
    centers = layer_center_depths(depths)
    obs_coords = depth_coords_1d(centers)
    obs_vp = model.vp.unsqueeze(-1)
    obs_vs = model.vs.unsqueeze(-1)
    grid = depth_coords_1d(
        torch.linspace(0.0, depths[-1], n_grid, device=depths.device)
    )
    vp_grid = hnf_field(obs_coords, obs_vp, grid).squeeze(-1)
    vs_grid = hnf_field(obs_coords, obs_vs, grid).squeeze(-1)
    vp_grad = vp_grid[1:] - vp_grid[:-1]
    vs_grad = vs_grid[1:] - vs_grid[:-1]
    smooth = torch.mean(vp_grad ** 2) + torch.mean(vs_grad ** 2)
    norm_centers = centers / depths[-1].clamp(min=1e-6)
    norm_grid = torch.linspace(0.0, 1.0, n_grid, device=depths.device)
    idx = torch.searchsorted(norm_grid, norm_centers).clamp(max=n_grid - 1)
    tie = torch.mean((model.vp - vp_grid[idx]) ** 2) + torch.mean((model.vs - vs_grid[idx]) ** 2)
    return smooth + tie


def _set_trainable(model: InvertibleLayeredEarth1D, vp_vs: bool, q: bool) -> None:
    model.log_vp_base.requires_grad_(vp_vs)
    model.log_vp_inc.requires_grad_(vp_vs)
    model.logit_vs_ratio.requires_grad_(vp_vs)
    if model.invert_q:
        model.log_q_base.requires_grad_(q)
        model.log_q_inc.requires_grad_(q)


def invert_layered_1d(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    receiver_distances: torch.Tensor,
    obs: dict[str, torch.Tensor],
    steps: int = 500,
    lr: float = 0.05,
    smooth_weight: float = 0.05,
    anchor_weight: float = 0.005,
    amp_weight: float = 0.0,
    q_anchor_weight: float = 0.002,
    invert_q: bool = False,
    frequency_hz: float | list[float] = 8.0,
    hnf_field=None,
    hnf_weight: float = 0.0,
    two_stage_q: bool = False,
    verbose: bool = False,
) -> tuple[InvertibleLayeredEarth1D, list[dict[str, float]]]:
    """Gradient-based inversion of layer vp/vs(/Q) from sparse observations."""
    if two_stage_q and amp_weight > 0 and "log_ap" in obs:
        model = InvertibleLayeredEarth1D(
            depths, vp_init, vs_init, q_init, invert_q=True
        )
        _set_trainable(model, vp_vs=True, q=False)
        hist_a = _invert_loop(
            model, source_depth, receiver_distances, obs, steps // 2, lr,
            vp_init, vs_init, q_init, smooth_weight, anchor_weight,
            amp_weight=0.0, q_anchor_weight=q_anchor_weight,
            frequency_hz=frequency_hz, hnf_field=hnf_field, hnf_weight=hnf_weight,
            times_weight=1.0, verbose=verbose,
        )
        _set_trainable(model, vp_vs=False, q=True)
        hist_b = _invert_loop(
            model, source_depth, receiver_distances, obs, steps - steps // 2, max(lr * 2.0, 0.1),
            vp_init, vs_init, q_init, 0.0, 0.0,
            amp_weight=amp_weight, q_anchor_weight=q_anchor_weight * 0.1,
            frequency_hz=frequency_hz, hnf_field=None, hnf_weight=0.0,
            times_weight=0.0, verbose=verbose,
        )
        return model, hist_a + hist_b

    model = InvertibleLayeredEarth1D(
        depths, vp_init, vs_init, q_init, invert_q=invert_q or amp_weight > 0
    )
    _set_trainable(model, vp_vs=True, q=invert_q or amp_weight > 0)
    history = _invert_loop(
        model, source_depth, receiver_distances, obs, steps, lr,
        vp_init, vs_init, q_init, smooth_weight, anchor_weight,
        amp_weight, q_anchor_weight, frequency_hz, hnf_field, hnf_weight,
        times_weight=1.0, verbose=verbose,
    )
    return model, history


def _invert_loop(
    model: InvertibleLayeredEarth1D,
    source_depth: float,
    receiver_distances: torch.Tensor,
    obs: dict[str, torch.Tensor],
    steps: int,
    lr: float,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    smooth_weight: float,
    anchor_weight: float,
    amp_weight: float,
    q_anchor_weight: float,
    frequency_hz: float | list[float],
    hnf_field,
    hnf_weight: float,
    times_weight: float,
    verbose: bool,
) -> list[dict[str, float]]:
    params = [p for p in model.parameters() if p.requires_grad]
    if hnf_field is not None and hnf_weight > 0:
        params += list(hnf_field.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    src = torch.tensor(source_depth, dtype=model.depths.dtype, device=model.depths.device)
    with_amp = amp_weight > 0 and "log_ap" in obs
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        opt.zero_grad()
        pred = model(src, receiver_distances, frequency_hz, with_amplitude=with_amp)
        hnf_reg = None
        if hnf_field is not None and hnf_weight > 0:
            hnf_reg = hnf_profile_regularization(model, hnf_field)
        loss, metrics = inversion_loss(
            pred,
            obs,
            model,
            vp_init,
            vs_init,
            q_init if model.invert_q else None,
            smooth_weight,
            anchor_weight,
            amp_weight,
            q_anchor_weight,
            hnf_reg,
            hnf_weight,
            times_weight=times_weight,
        )
        loss.backward()
        opt.step()
        history.append(metrics)
        if verbose and (step == 1 or step % max(1, steps // 10) == 0 or step == steps):
            msg = (
                f"[inv] step {step}/{steps} loss={metrics['loss']:.6f} "
                f"tp={metrics['loss_tp']:.6f} ts={metrics['loss_ts']:.6f}"
            )
            if "loss_amp_p" in metrics:
                msg += f" amp_p={metrics['loss_amp_p']:.6f} amp_s={metrics['loss_amp_s']:.6f}"
            if "loss_hnf" in metrics:
                msg += f" hnf={metrics['loss_hnf']:.6f}"
            print(msg, flush=True)
    return history


def model_rmse(true: LayeredEarth1D, recovered: LayeredEarth1D) -> dict[str, float]:
    return {
        "vp_rmse": float(torch.sqrt(torch.mean((true.vp - recovered.vp) ** 2))),
        "vs_rmse": float(torch.sqrt(torch.mean((true.vs - recovered.vs) ** 2))),
        "q_rmse": float(torch.sqrt(torch.mean((true.q - recovered.q) ** 2))),
    }


def default_synth_model(device: torch.device | str = "cpu") -> LayeredEarth1D:
    """Canonical 5-layer synthetic model (km, km/s)."""
    depths = torch.tensor([0.0, 2.0, 6.0, 12.0, 20.0, 35.0], device=device)
    vp = torch.tensor([3.5, 4.5, 5.5, 6.2, 6.8], device=device)
    vs = torch.tensor([2.0, 2.6, 3.2, 3.6, 3.9], device=device)
    q = torch.tensor([80.0, 120.0, 150.0, 200.0, 250.0], device=device)
    return LayeredEarth1D(depths=depths, vp=vp, vs=vs, q=q)


def default_station_distances(device: torch.device | str = "cpu", n_stations: int = 8) -> torch.Tensor:
    """Sparse surface receivers (km), including zero offset."""
    if n_stations < 2:
        return torch.tensor([0.0], device=device)
    dmax = 50.0
    return torch.linspace(0.0, dmax, n_stations, device=device)
