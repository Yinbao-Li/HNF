# -*- coding: utf-8 -*-
"""
FWI-lite: differentiable direct-wave synthesis + optional 2D FD forward.

Inversion uses ray-travel-time-linked Ricker waveforms (fully differentiable).
FD engine is kept for visualization / non-gradient forward checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from hnf.inversion_1d import (
    InvertibleLayeredEarth1D,
    LayeredEarth1D,
    layer_index_at_depth,
    model_rmse,
    travel_time_phase,
)


def ricker_source(t: torch.Tensor, f0: float, t0: torch.Tensor | float) -> torch.Tensor:
    """Ricker wavelet; supports broadcast between t (1,nt) and t0 (n,1)."""
    if isinstance(t0, (float, int)):
        t0 = torch.tensor(t0, dtype=t.dtype, device=t.device)
    x = (torch.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * x) * torch.exp(-x)


def layer_velocity_to_grid(
    depths: torch.Tensor,
    vp_layers: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    """Map 1D layer vp to depth grid."""
    idx = layer_index_at_depth(depths, z_grid)
    return vp_layers[idx]


@dataclass
class FWIResult:
    name: str
    earth: LayeredEarth1D
    history: list[dict[str, float]]
    waveform_misfit: float
    rmse: dict[str, float]
    wall_sec: float


class DirectWaveForward:
    """
    Differentiable multi-station direct P/S waveforms from layered ray travel times.

    This is the default FWI-lite engine: waveform misfit back-propagates to vp/vs
  through travel_time_phase().
    """

    def __init__(
        self,
        dt: float = 0.01,
        nt: int = 1200,
        f0_p: float = 2.5,
        f0_s: float = 1.5,
        amp_p: float = 1.0,
        amp_s: float = 0.65,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.dt = dt
        self.nt = nt
        self.f0_p = f0_p
        self.f0_s = f0_s
        self.amp_p = amp_p
        self.amp_s = amp_s
        self.time = torch.arange(nt, device=self.device, dtype=torch.float32) * dt

    def simulate(
        self,
        earth: LayeredEarth1D,
        source_depth: float,
        receiver_x: torch.Tensor,
    ) -> torch.Tensor:
        """Return seismograms (n_recv, nt)."""
        dev = earth.vp.device
        src = torch.tensor(source_depth, dtype=earth.vp.dtype, device=dev)
        rx = receiver_x.to(dev)
        tp = travel_time_phase(earth, "P", src, rx)
        ts = travel_time_phase(earth, "S", src, rx)
        t = self.time.view(1, -1)
        wp = ricker_source(t, self.f0_p, tp.view(-1, 1))
        ws = ricker_source(t, self.f0_s, ts.view(-1, 1))
        return self.amp_p * wp + self.amp_s * ws


def project_layered_velocities(vp: torch.Tensor, vs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp to simple physically valid 1D layered model constraints."""
    vp_proj = vp.clamp(min=1.5)
    vp_parts = [vp_proj[:1]]
    for i in range(1, vp_proj.numel()):
        vp_parts.append(torch.maximum(vp_proj[i : i + 1], vp_parts[-1] + 0.05))
    vp_proj = torch.cat(vp_parts)

    vs_proj = vs.clamp(min=1.0)
    vs_parts = [torch.minimum(vs_proj[:1], vp_proj[:1] * 0.75)]
    for i in range(1, vs_proj.numel()):
        upper = vp_proj[i : i + 1] * 0.75
        vs_parts.append(torch.minimum(vs_proj[i : i + 1], upper))
    vs_proj = torch.cat(vs_parts)
    return vp_proj, vs_proj


def unrolled_waveform_refine(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q: torch.Tensor,
    source_depth: float,
    receiver_distances: torch.Tensor,
    observed: torch.Tensor,
    steps: int = 5,
    step_size: float = 0.05,
    anchor_weight: float = 0.005,
    smooth_weight: float = 0.05,
    dt: float = 0.01,
) -> tuple[LayeredEarth1D, dict[str, torch.Tensor]]:
    """
    Differentiable short-step waveform refinement.

    Unlike `invert_acoustic_fwi`, this keeps the inner updates differentiable with
    respect to the initialization so it can be used as a training objective.
    """
    dev = depths.device
    engine = DirectWaveForward(device=dev, nt=observed.shape[-1], dt=dt)
    vp = vp_init
    vs = vs_init
    metrics: dict[str, torch.Tensor] = {}

    for _ in range(steps):
        vp = vp.clone().requires_grad_(True)
        vs = vs.clone().requires_grad_(True)
        earth = LayeredEarth1D(depths=depths, vp=vp, vs=vs, q=q)
        pred = engine.simulate(earth, source_depth, receiver_distances)
        loss_w = torch.mean((pred - observed) ** 2)
        smooth = torch.mean((vp[1:] - vp[:-1]) ** 2)
        anchor = torch.mean((vp - vp_init) ** 2) + torch.mean((vs - vs_init) ** 2)
        loss = loss_w + smooth_weight * smooth + anchor_weight * anchor
        g_vp, g_vs = torch.autograd.grad(loss, [vp, vs], create_graph=True)
        vp = vp - step_size * g_vp
        vs = vs - step_size * g_vs
        vp, vs = project_layered_velocities(vp, vs)
        metrics = {
            "loss": loss,
            "waveform": loss_w,
            "smooth": smooth,
            "anchor": anchor,
        }

    earth = LayeredEarth1D(depths=depths, vp=vp, vs=vs, q=q)
    return earth, metrics


class Acoustic2DForward:
    """Constant-density 2D acoustic FD (forward-only, no autograd through time loop)."""

    def __init__(
        self,
        x_max: float = 50.0,
        z_max: float = 35.0,
        nx: int = 51,
        nz: int = 36,
        dt: float = 0.012,
        nt: int = 900,
        f0: float = 2.0,
        t0: float = 0.15,
        sponge_width: int = 8,
        sponge_strength: float = 0.25,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.dt = dt
        self.f0 = f0
        self.t0 = t0
        self.dx = x_max / (nx - 1)
        self.dz = z_max / (nz - 1)
        self.x = torch.linspace(0.0, x_max, nx, device=self.device)
        self.z = torch.linspace(0.0, z_max, nz, device=self.device)
        self.time = torch.arange(nt, device=self.device, dtype=torch.float32) * dt
        self.src_amp = ricker_source(self.time, f0, t0)
        self.sponge = self._build_sponge(sponge_width, sponge_strength)

    def _build_sponge(self, width: int, strength: float) -> torch.Tensor:
        damp = torch.zeros(self.nz, self.nx, device=self.device)
        for w in range(width):
            coef = strength * (w + 1) / width
            damp[w, :] = torch.maximum(damp[w, :], torch.tensor(coef, device=self.device))
            damp[-w - 1, :] = torch.maximum(damp[-w - 1, :], torch.tensor(coef, device=self.device))
            damp[:, w] = torch.maximum(damp[:, w], torch.tensor(coef, device=self.device))
            damp[:, -w - 1] = torch.maximum(damp[:, -w - 1], torch.tensor(coef, device=self.device))
        return damp

    def _laplacian(self, u: torch.Tensor) -> torch.Tensor:
        u_pad = torch.zeros(self.nz + 2, self.nx + 2, device=u.device, dtype=u.dtype)
        u_pad[1:-1, 1:-1] = u
        lap = (
            (u_pad[1:-1, 2:] - 2 * u + u_pad[1:-1, :-2]) / (self.dx ** 2)
            + (u_pad[2:, 1:-1] - 2 * u + u_pad[:-2, 1:-1]) / (self.dz ** 2)
        )
        return lap

    @torch.no_grad()
    def simulate(
        self,
        vp_grid: torch.Tensor,
        source_depth: float,
        receiver_x: torch.Tensor,
    ) -> torch.Tensor:
        v2 = vp_grid ** 2
        v2_2d = v2.unsqueeze(1).expand(self.nz, self.nx)
        src_z = int(round(source_depth / self.dz))
        src_z = min(max(src_z, 1), self.nz - 2)
        src_x = 0
        rx_idx = (receiver_x / self.dx).round().long().clamp(0, self.nx - 1)

        u = torch.zeros(self.nz, self.nx, device=self.device)
        u_prev = torch.zeros_like(u)
        records = []

        for n in range(self.nt):
            lap = self._laplacian(u)
            u_next = 2 * u - u_prev + (self.dt ** 2) * v2_2d * lap
            u_next[src_z, src_x] = u_next[src_z, src_x] + self.src_amp[n]
            u_next = u_next * (1.0 - self.sponge)
            u_prev = u
            u = u_next
            records.append(u[0, rx_idx].clone())

        return torch.stack(records, dim=1)


def invert_acoustic_fwi(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    true_model: LayeredEarth1D,
    source_depth: float,
    receiver_distances: torch.Tensor,
    observed: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    anchor_weight: float = 0.005,
    smooth_weight: float = 0.05,
    verbose: bool = False,
) -> tuple[InvertibleLayeredEarth1D, list[dict[str, float]], DirectWaveForward]:
    """Invert vp/vs from waveform misfit via differentiable direct-wave forward."""
    dev = depths.device
    engine = DirectWaveForward(device=dev, nt=observed.shape[-1], dt=0.01)
    model = InvertibleLayeredEarth1D(depths, vp_init, vs_init, q_init, invert_q=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        opt.zero_grad()
        pred = engine.simulate(model.earth, source_depth, receiver_distances)
        loss_w = torch.mean((pred - observed) ** 2)
        smooth = torch.mean(torch.exp(model.log_vp_inc) ** 2)
        anchor = torch.mean((model.vp - vp_init) ** 2) + torch.mean((model.vs - vs_init) ** 2)
        loss = loss_w + smooth_weight * smooth + anchor_weight * anchor
        loss.backward()
        opt.step()
        history.append({
            "loss": float(loss.detach()),
            "waveform": float(loss_w.detach()),
            "smooth": float(smooth.detach()),
            "anchor": float(anchor.detach()),
        })
        if verbose and (step == 1 or step % max(1, steps // 10) == 0 or step == steps):
            print(
                f"[fwi] step {step}/{steps} loss={history[-1]['loss']:.6f} "
                f"wave={history[-1]['waveform']:.6f}",
                flush=True,
            )

    return model, history, engine


def run_fwi_lite_baseline(
    true_model: LayeredEarth1D,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    receiver_distances: torch.Tensor,
    noise_std: float = 0.02,
    steps: int = 200,
    seed: int = 42,
) -> FWIResult:
    t0 = time.perf_counter()
    dev = true_model.vp.device
    engine = DirectWaveForward(device=dev)
    clean = engine.simulate(true_model, source_depth, receiver_distances)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    observed = clean + noise_std * torch.randn(clean.shape, generator=gen, device=dev)

    model, history, _ = invert_acoustic_fwi(
        true_model.depths, vp_init, vs_init, q_init,
        true_model, source_depth, receiver_distances, observed,
        steps=steps, verbose=False,
    )
    rec = model.earth
    with torch.no_grad():
        pred = engine.simulate(rec, source_depth, receiver_distances)
        wf_misfit = float(torch.mean((pred - observed) ** 2))
    return FWIResult(
        name="FWI-lite (direct-wave)",
        earth=rec,
        history=history,
        waveform_misfit=wf_misfit,
        rmse=model_rmse(true_model, rec),
        wall_sec=time.perf_counter() - t0,
    )
