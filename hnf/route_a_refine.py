# -*- coding: utf-8 -*-
"""
Route A: Zhizi velocity prior + damped Gauss-Newton travel-time refinement.

Answers: does the velocity from Zhizi carry physical meaning as a GN initializer?
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hnf.inversion_1d import LayeredEarth1D, model_rmse
from hnf.inversion_baselines import InversionResult, invert_gauss_newton


@dataclass
class RouteARow:
    idx: int
    zhizi_init_vp_rmse: float
    perturb_init_vp_rmse: float
    zhizi_refined_vp_rmse: float
    perturb_refined_vp_rmse: float
    zhizi_init_tt: float
    perturb_init_tt: float
    zhizi_refined_tt: float
    perturb_refined_tt: float
    zhizi_refine_sec: float
    perturb_refine_sec: float
    gn_iters: int


@dataclass
class RouteAVerdict:
    n_events: int
    gn_iters: int
    init_rmse_ratio: float
    refined_rmse_ratio: float
    init_zhizi_better_frac: float
    refined_zhizi_better_frac: float
    convergence_at_5_ratio: float | None
    physically_meaningful: bool
    rationale: str


def rmse_vs_true(true_earth: LayeredEarth1D, earth: LayeredEarth1D) -> dict[str, float]:
    return model_rmse(true_earth, earth)


def refine_gn(
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
    return invert_gauss_newton(
        depths, vp_init, vs_init, q_init,
        source_depth, distances, obs_tp, obs_ts,
        n_iter=n_iter, damping=damping,
    )


def build_verdict(rows: list[RouteARow], conv5_rows: list[RouteARow] | None = None) -> RouteAVerdict:
    """Heuristic verdict for Route A physical-meaning test."""
    n = max(len(rows), 1)
    init_ratio = sum(r.zhizi_init_vp_rmse / max(r.perturb_init_vp_rmse, 1e-6) for r in rows) / n
    refined_ratio = sum(r.zhizi_refined_vp_rmse / max(r.perturb_refined_vp_rmse, 1e-6) for r in rows) / n
    init_better = sum(1 for r in rows if r.zhizi_init_vp_rmse < r.perturb_init_vp_rmse) / n
    refined_better = sum(1 for r in rows if r.zhizi_refined_vp_rmse < r.perturb_refined_vp_rmse) / n

    conv5_ratio = None
    if conv5_rows:
        m = max(len(conv5_rows), 1)
        conv5_ratio = sum(
            r.zhizi_refined_vp_rmse / max(r.perturb_refined_vp_rmse, 1e-6) for r in conv5_rows
        ) / m

    gn_iters = rows[0].gn_iters if rows else 25

    meaningful = False
    reasons: list[str] = []

    if init_ratio < 0.85:
        meaningful = True
        reasons.append(f"智子初值 Vp RMSE 均值仅为扰动初值的 {init_ratio:.2f}×")
    if init_better >= 0.6:
        meaningful = True
        reasons.append(f"{init_better:.0%} 事件上智子初值优于扰动初值")
    if conv5_ratio is not None and conv5_ratio < 0.95:
        meaningful = True
        reasons.append(f"5 步 GN 后智子路径 Vp RMSE 为扰动路径的 {conv5_ratio:.2f}×（收敛更快）")
    if refined_ratio > 1.15:
        meaningful = False
        reasons.append(f"全收敛后智子+GN 反而更差 ({refined_ratio:.2f}×)，初值可能误导优化")
    if not meaningful and not reasons:
        reasons.append(
            f"智子初值未显著优于扰动 (ratio={init_ratio:.2f})，"
            f"全收敛后二者接近 (ratio={refined_ratio:.2f})"
        )

    return RouteAVerdict(
        n_events=len(rows),
        gn_iters=gn_iters,
        init_rmse_ratio=init_ratio,
        refined_rmse_ratio=refined_ratio,
        init_zhizi_better_frac=init_better,
        refined_zhizi_better_frac=refined_better,
        convergence_at_5_ratio=conv5_ratio,
        physically_meaningful=meaningful,
        rationale="; ".join(reasons),
    )
