# -*- coding: utf-8 -*-
"""Classical 1D travel-time inversion baselines for comparison with HNF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from hnf.inversion_1d import (
    InvertibleLayeredEarth1D,
    LayeredEarth1D,
    invert_layered_1d,
    model_rmse,
    synthesize_travel_times,
    travel_time_phase,
)


@dataclass
class InversionResult:
    name: str
    earth: LayeredEarth1D
    history: list[dict[str, float]]
    time_misfit: float
    rmse: dict[str, float]
    wall_sec: float = 0.0


def _time_misfit(
    model: LayeredEarth1D,
    source_depth: float,
    distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
) -> float:
    src = torch.tensor(source_depth, dtype=model.vp.dtype, device=model.vp.device)
    tp = travel_time_phase(model, "P", src, distances)
    ts = travel_time_phase(model, "S", src, distances)
    return float(torch.mean((tp - obs_tp) ** 2) + torch.mean((ts - obs_ts) ** 2))


def invert_hnf_adam(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs: dict[str, torch.Tensor],
    steps: int = 800,
    lr: float = 0.06,
) -> InversionResult:
    import time

    t0 = time.perf_counter()
    model, history = invert_layered_1d(
        depths, vp_init, vs_init, q_init, source_depth, distances, obs,
        steps=steps, lr=lr, verbose=False,
    )
    earth = model.earth
    return InversionResult(
        name="HNF-Adam",
        earth=earth,
        history=history,
        time_misfit=_time_misfit(earth, source_depth, distances, obs["tp"], obs["ts"]),
        rmse={},
        wall_sec=time.perf_counter() - t0,
    )


def invert_lbfgs_torch(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
    max_iter: int = 80,
) -> InversionResult:
    """L-BFGS on the same monotonic HNF parameterization."""
    import time

    t0 = time.perf_counter()
    model = InvertibleLayeredEarth1D(depths, vp_init, vs_init, q_init, invert_q=False)
    src = torch.tensor(source_depth, dtype=depths.dtype, device=depths.device)
    obs = {"tp": obs_tp, "ts": obs_ts}
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    history: list[dict[str, float]] = []

    def closure():
        opt.zero_grad()
        pred = model(src, distances)
        loss_tp = torch.mean((pred["tp"] - obs_tp) ** 2)
        loss_ts = torch.mean((pred["ts"] - obs_ts) ** 2)
        smooth = torch.mean(torch.exp(model.log_vp_inc) ** 2)
        anchor = torch.mean((model.vp - vp_init) ** 2) + torch.mean((model.vs - vs_init) ** 2)
        loss = loss_tp + loss_ts + 0.05 * smooth + 0.005 * anchor
        loss.backward()
        history.append({
            "loss": float(loss.detach()),
            "loss_tp": float(loss_tp.detach()),
            "loss_ts": float(loss_ts.detach()),
        })
        return loss

    opt.step(closure)
    earth = model.earth
    return InversionResult(
        name="L-BFGS (PyTorch)",
        earth=earth,
        history=history,
        time_misfit=_time_misfit(earth, source_depth, distances, obs_tp, obs_ts),
        rmse={},
        wall_sec=time.perf_counter() - t0,
    )


def invert_gauss_newton(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
    n_iter: int = 25,
    damping: float = 0.1,
) -> InversionResult:
    """
    Damped Gauss-Newton on layer vp/vs (log-space), same ray forward model.
    Classic tomography-style linearized update each iteration.
    """
    import time

    t0 = time.perf_counter()
    dev = depths.device
    vp = vp_init.clone().to(dev)
    vs = vs_init.clone().to(dev)
    q = q_init.clone().to(dev)
    distances = distances.to(dev)
    obs_tp = obs_tp.to(dev)
    obs_ts = obs_ts.to(dev)
    history: list[dict[str, float]] = []
    n = vp.numel()

    for _ in range(n_iter):
        earth = LayeredEarth1D(depths=depths, vp=vp, vs=vs, q=q)
        src = torch.tensor(source_depth, dtype=vp.dtype, device=dev)
        tp = travel_time_phase(earth, "P", src, distances)
        ts = travel_time_phase(earth, "S", src, distances)
        r = torch.cat([tp - obs_tp, ts - obs_ts])
        misfit = float(torch.mean(r * r))
        history.append({"loss": misfit, "loss_tp": float(torch.mean((tp - obs_tp) ** 2)), "loss_ts": float(torch.mean((ts - obs_ts) ** 2))})

        J_cols = []
        for i in range(n):
            dvp = torch.zeros_like(vp)
            dvp[i] = 1e-3 * vp[i].clamp(min=1.0)
            e_p = LayeredEarth1D(depths, vp + dvp, vs, q)
            tp_p = travel_time_phase(e_p, "P", src, distances)
            ts_p = travel_time_phase(e_p, "S", src, distances)
            J_cols.append(torch.cat([(tp_p - tp) / dvp[i], (ts_p - ts) / dvp[i]]))
        for i in range(n):
            dvs = torch.zeros_like(vs)
            dvs[i] = 1e-3 * vs[i].clamp(min=1.0)
            e_s = LayeredEarth1D(depths, vp, vs + dvs, q)
            tp_s = travel_time_phase(e_s, "P", src, distances)
            ts_s = travel_time_phase(e_s, "S", src, distances)
            J_cols.append(torch.cat([(tp_s - tp) / dvs[i], (ts_s - ts) / dvs[i]]))
        J = torch.stack(J_cols, dim=1)
        JTJ = J.T @ J + damping * torch.eye(2 * n, dtype=J.dtype, device=J.device)
        delta = torch.linalg.solve(JTJ, -J.T @ r)
        vp = (vp + delta[:n]).clamp(min=1.5)
        vs = (vs + delta[n:]).clamp(min=1.0)
        for i in range(1, n):
            vp[i] = torch.maximum(vp[i], vp[i - 1] + 0.02)
            vs[i] = torch.minimum(vs[i], vp[i] * 0.75)

    earth = LayeredEarth1D(depths=depths, vp=vp, vs=vs, q=q)
    return InversionResult(
        name="Gauss-Newton (damped)",
        earth=earth,
        history=history,
        time_misfit=_time_misfit(earth, source_depth, distances, obs_tp, obs_ts),
        rmse={},
        wall_sec=time.perf_counter() - t0,
    )


def invert_homogeneous_grid(
    depths: torch.Tensor,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs_tp: torch.Tensor,
    obs_ts: torch.Tensor,
    n_grid: int = 60,
) -> InversionResult:
    """Brute-force 1D grid on uniform vp/vs (no layering) — naive baseline."""
    import time

    t0 = time.perf_counter()
    n_layers = depths.numel() - 1
    vp_grid = torch.linspace(2.5, 8.5, n_grid, device=depths.device)
    vs_grid = torch.linspace(1.5, 5.0, n_grid, device=depths.device)
    src = torch.tensor(source_depth, dtype=vp_init.dtype, device=depths.device)
    distances = distances.to(depths.device)
    obs_tp = obs_tp.to(depths.device)
    obs_ts = obs_ts.to(depths.device)
    best_loss = float("inf")
    best_vp, best_vs = vp_init.clone(), vs_init.clone()
    history: list[dict[str, float]] = []

    for vp0 in vp_grid:
        for vs0 in vs_grid:
            if vs0 >= vp0 * 0.8:
                continue
            vp = torch.full((n_layers,), float(vp0))
            vs = torch.full((n_layers,), float(vs0))
            earth = LayeredEarth1D(depths, vp, vs, q_init)
            loss = _time_misfit(earth, source_depth, distances, obs_tp, obs_ts)
            if loss < best_loss:
                best_loss = loss
                best_vp, best_vs = vp, vs
    history.append({"loss": best_loss})
    earth = LayeredEarth1D(depths, best_vp, best_vs, q_init)
    return InversionResult(
        name="Homogeneous grid",
        earth=earth,
        history=history,
        time_misfit=best_loss,
        rmse={},
        wall_sec=time.perf_counter() - t0,
    )


def run_all_baselines(
    true_model: LayeredEarth1D,
    vp_init: torch.Tensor,
    vs_init: torch.Tensor,
    q_init: torch.Tensor,
    source_depth: float,
    distances: torch.Tensor,
    obs: dict[str, torch.Tensor],
    steps: int = 800,
) -> list[InversionResult]:
    runners: list[Callable[[], InversionResult]] = [
        lambda: invert_hnf_adam(
            true_model.depths, vp_init, vs_init, q_init,
            source_depth, distances, obs, steps=steps,
        ),
        lambda: invert_lbfgs_torch(
            true_model.depths, vp_init, vs_init, q_init,
            source_depth, distances, obs["tp"], obs["ts"],
        ),
        lambda: invert_gauss_newton(
            true_model.depths, vp_init, vs_init, q_init,
            source_depth, distances, obs["tp"], obs["ts"],
        ),
        lambda: invert_homogeneous_grid(
            true_model.depths, vp_init, vs_init, q_init,
            source_depth, distances, obs["tp"], obs["ts"],
        ),
    ]
    results = []
    for fn in runners:
        res = fn()
        res.rmse = model_rmse(true_model, res.earth)
        results.append(res)
    return results
