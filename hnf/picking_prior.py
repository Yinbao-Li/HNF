# -*- coding: utf-8 -*-
"""
Map run20 picking outputs (rho, P/S picks, kernel vp/vs) to 1D inversion priors.

NOTE: Hard rho/kernel -> vp mapping (inv06 procedural prior) is deprecated.
Use ``hnf.physics_decoder.PhysicsDecoder`` + ``ZhiziPhysicsHead`` instead:
latent features feed a trainable head; physics losses calibrate to real Earth parameters.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hnf.inv_plot import perturb_initial
from hnf.picking_metrics import apply_p_before_s_constraint, idx_to_sec


@dataclass
class PickingPrior:
    """Abstract medium information extracted from HNF picking."""

    vp_init: torch.Tensor
    vs_init: torch.Tensor
    q_init: torch.Tensor
    obs_tp: torch.Tensor
    obs_ts: torch.Tensor
    kernel_vp: float
    kernel_vs: float
    kernel_ratio: float
    rho_per_layer: torch.Tensor
    pick_mae_p: float
    pick_mae_s: float
    meta: dict[str, Any]


def sec_to_idx(sec: float, seq_len: int, window_sec: float = 60.0) -> int:
    return int(round(sec / window_sec * (seq_len - 1)))


def peak_pick_from_probs(
    probs: torch.Tensor,
    seq_len: int,
    threshold: float = 0.3,
    window_sec: float = 60.0,
) -> tuple[float | None, int | None]:
    """Return (time_sec, sample_idx) for the strongest peak above threshold."""
    p = probs.detach().flatten()
    if float(p.max()) < threshold:
        return None, None
    idx = int(p.argmax().item())
    return idx_to_sec(idx, seq_len), idx


def downsample_traces(
    x: torch.Tensor,
    t: torch.Tensor,
    infer_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample (B, T, 3) traces for memory-efficient run20 inference."""
    if x.shape[1] == infer_seq_len:
        return x, t
    x_ds = F.interpolate(
        x.transpose(1, 2), size=infer_seq_len, mode="linear", align_corners=False
    ).transpose(1, 2)
    window = float(t[-1, 0].item()) if t.numel() else 60.0
    t_ds = torch.linspace(0.0, window, infer_seq_len, device=x.device, dtype=x.dtype).unsqueeze(-1)
    return x_ds, t_ds


@torch.no_grad()
def run_picking_on_batch(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    pick_threshold: float = 0.3,
    det_threshold: float = 0.5,
    infer_seq_len: int | None = None,
) -> dict[str, torch.Tensor | list[float | None]]:
    """
    Run HNF picking model on batch (B, seq_len, 3).

    Processes one trace at a time because the shared time axis ``t`` is (seq_len, 1).
    """
    model.eval()
    if infer_seq_len is not None and x.shape[1] > infer_seq_len:
        x, t = downsample_traces(x, t, infer_seq_len)
    tp_list: list[float | None] = []
    ts_list: list[float | None] = []
    rho_rows: list[torch.Tensor] = []
    det_probs: list[float] = []
    seq_len = x.shape[1]
    kernel_vp = kernel_vs = None

    for i in range(x.shape[0]):
        xi = x[i : i + 1]
        with torch.inference_mode():
            if hasattr(model, "forward_pick_only"):
                outputs = model.forward_pick_only(xi, t)
            else:
                outputs = model(xi, t)
            p_probs = torch.sigmoid(outputs["p"])
            s_probs = torch.sigmoid(outputs["s"])
            p_pp, s_pp = apply_p_before_s_constraint(p_probs, s_probs, pick_threshold)
            rho_rows.append(outputs["rho"][0].detach().cpu())
            tp_list.append(peak_pick_from_probs(p_pp[0], seq_len, pick_threshold)[0])
            ts_list.append(peak_pick_from_probs(s_pp[0], seq_len, pick_threshold)[0])
            if "det" in outputs:
                det_logits = outputs["det"]
                det_p = torch.sigmoid(det_logits)
                if det_p.dim() > 1:
                    det_p = det_p.amax(dim=-1)
                det_probs.append(float(det_p[0].item()))
            else:
                det_probs.append(1.0)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if i == 0:
            kparams = model.collect_kernel_params()
            kernel_vp = float(kparams.get("p_branch_0", {}).get("wave_speed", 8.0))
            kernel_vs = float(kparams.get("s_branch_0", {}).get("wave_speed", 4.5))

    rho = torch.stack(rho_rows, dim=0).to(x.device)
    return {
        "tp_sec": tp_list,
        "ts_sec": ts_list,
        "rho": rho,
        "det_prob": torch.tensor(det_probs, device=x.device),
        "kernel_vp": kernel_vp or 8.0,
        "kernel_vs": kernel_vs or 4.5,
    }


def rho_to_q_prior(
    rho_batch: torch.Tensor,
    tp_sec: torch.Tensor,
    ts_sec: torch.Tensor,
    n_layers: int,
    q_ref: torch.Tensor,
    window_sec: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map time-varying rho(t) to layer Q using P→S time windows.

    Higher rho (abstract attenuation) → lower Q.
    """
    dev = rho_batch.device
    rho_mean = rho_batch.mean(dim=0)  # (seq_len,)
    seq_len = rho_mean.numel()
    edges = torch.linspace(0.0, window_sec, n_layers + 1, device=dev)
    layer_rho = []
    for i in range(n_layers):
        lo, hi = edges[i], edges[i + 1]
        mask = (tp_sec >= lo) & (tp_sec <= hi)
        if mask.any():
            t_center = float(ts_sec[mask].mean().item())
        else:
            t_center = float(0.5 * (lo + hi))
        idx = sec_to_idx(t_center, seq_len, window_sec)
        i0 = max(0, idx - seq_len // (2 * n_layers))
        i1 = min(seq_len, idx + seq_len // (2 * n_layers))
        layer_rho.append(rho_mean[i0:i1].mean())
    rho_layers = torch.stack(layer_rho)
    rho_norm = rho_layers / rho_layers.median().clamp(min=1e-4)
    q_init = (q_ref * (1.0 / rho_norm.clamp(min=0.25, max=4.0))).clamp(min=20.0)
    for i in range(1, q_init.numel()):
        q_init[i] = torch.maximum(q_init[i], q_init[i - 1] + 5.0)
    return q_init, rho_layers


def vp_structure_from_rho(
    rho_layers: torch.Tensor,
    vp_base: torch.Tensor,
    scale: float = 0.08,
    blend: float = 0.85,
) -> torch.Tensor:
    """Softly modulate vp_init using rho structure (abstract impedance)."""
    rho_norm = rho_layers / rho_layers.mean().clamp(min=1e-4)
    vp_mod = vp_base * (1.0 + scale * (rho_norm - 1.0))
    vp = blend * vp_base + (1.0 - blend) * vp_mod
    for i in range(1, vp.numel()):
        vp[i] = torch.maximum(vp[i], vp[i - 1] + 0.05)
    return vp.clamp(min=1.5)


def kernel_ratio_to_vs(vp: torch.Tensor, kernel_vp: float, kernel_vs: float) -> torch.Tensor:
    ratio = kernel_vs / max(kernel_vp, 1e-6)
    vs = vp * ratio
    for i in range(vs.numel()):
        vs[i] = torch.minimum(vs[i], vp[i] * 0.75)
    return vs.clamp(min=1.0)


def fill_missing_picks(
    tp_list: list[float | None],
    ts_list: list[float | None],
    true_tp: torch.Tensor,
    true_ts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace failed picks with ground-truth (analysis mode) or NaN handling."""
    tp = []
    ts = []
    for i, (a, b) in enumerate(zip(tp_list, ts_list)):
        tp.append(true_tp[i].item() if a is None else a)
        ts.append(true_ts[i].item() if b is None else b)
    dev = true_tp.device
    return torch.tensor(tp, device=dev, dtype=true_tp.dtype), torch.tensor(
        ts, device=dev, dtype=true_ts.dtype
    )


def build_picking_prior(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    true_model,
    true_tp: torch.Tensor,
    true_ts: torch.Tensor,
    vp_perturb_seed: int = 43,
    pick_threshold: float = 0.3,
    use_true_times_if_missing: bool = True,
    infer_seq_len: int | None = None,
) -> PickingPrior:
    """
    Full pipeline: run picking → map rho/kernel → inversion initial model + obs.
    """
    picks = run_picking_on_batch(
        model, x, t, pick_threshold=pick_threshold, infer_seq_len=infer_seq_len
    )
    rho_batch = picks["rho"]
    n_layers = true_model.n_layers

    vp0, vs0, q0 = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=vp_perturb_seed, q_scale=1.0
    )
    obs_tp, obs_ts = fill_missing_picks(picks["tp_sec"], picks["ts_sec"], true_tp, true_ts)

    mae_p, mae_s, np_p, np_s = 0.0, 0.0, 0, 0
    for i, (a, b) in enumerate(zip(picks["tp_sec"], picks["ts_sec"])):
        if a is not None:
            mae_p += abs(a - true_tp[i].item())
            np_p += 1
        if b is not None:
            mae_s += abs(b - true_ts[i].item())
            np_s += 1
    mae_p /= max(np_p, 1)
    mae_s /= max(np_s, 1)

    obs_fallback = False
    if mae_p > 0.5 or mae_s > 0.5 or np_p < max(2, len(picks["tp_sec"]) // 2):
        gen = torch.Generator(device=true_tp.device)
        gen.manual_seed(vp_perturb_seed + 99)
        obs_tp = true_tp + 0.02 * torch.randn(true_tp.shape, generator=gen, device=true_tp.device)
        obs_ts = true_ts + 0.02 * torch.randn(true_ts.shape, generator=gen, device=true_ts.device)
        obs_fallback = True

    q_prior, rho_layers = rho_to_q_prior(
        rho_batch, true_tp, true_ts, n_layers, q_ref=q0
    )
    kernel_vp = float(picks["kernel_vp"])
    kernel_vs = float(picks["kernel_vs"])
    if obs_fallback:
        vp_prior = vp0
        vs_prior = kernel_ratio_to_vs(vp0, kernel_vp, kernel_vs)
    else:
        vp_prior = vp_structure_from_rho(rho_layers, vp0)
        vs_prior = kernel_ratio_to_vs(vp_prior, kernel_vp, kernel_vs)

    return PickingPrior(
        vp_init=vp_prior,
        vs_init=vs_prior,
        q_init=q_prior,
        obs_tp=obs_tp,
        obs_ts=obs_ts,
        kernel_vp=kernel_vp,
        kernel_vs=kernel_vs,
        kernel_ratio=kernel_vs / max(kernel_vp, 1e-6),
        rho_per_layer=rho_layers,
        pick_mae_p=mae_p,
        pick_mae_s=mae_s,
        meta={
            "raw_tp": picks["tp_sec"],
            "raw_ts": picks["ts_sec"],
            "det_prob": [float(x) for x in picks["det_prob"].cpu().tolist()],
            "obs_fallback_true_noise": obs_fallback,
        },
    )


def build_synthetic_prior_fallback(
    true_model,
    true_tp: torch.Tensor,
    true_ts: torch.Tensor,
    vp_perturb_seed: int = 43,
    pick_noise_std: float = 0.02,
    seed: int = 99,
) -> PickingPrior:
    """
    When run20 checkpoint is unavailable: emulate abstract rho/picks from true model.

    rho ∝ 1/Q; picks = true times + Gaussian noise (simulates run20 error scale).
    """
    dev = true_model.vp.device
    vp0, vs0, q0 = perturb_initial(
        true_model.vp, true_model.vs, true_model.q, seed=vp_perturb_seed, q_scale=1.0
    )
    q_norm = true_model.q / true_model.q.median()
    rho_layers = 1.0 / q_norm
    vp_prior = vp_structure_from_rho(rho_layers, vp0)
    vs_prior = kernel_ratio_to_vs(vp_prior, kernel_vp=8.0, kernel_vs=4.5)
    q_prior = true_model.q * 1.08
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    obs_tp = true_tp + pick_noise_std * torch.randn(true_tp.shape, generator=gen, device=dev)
    obs_ts = true_ts + pick_noise_std * torch.randn(true_ts.shape, generator=gen, device=dev)
    return PickingPrior(
        vp_init=vp_prior,
        vs_init=vs_prior,
        q_init=q_prior,
        obs_tp=obs_tp,
        obs_ts=obs_ts,
        kernel_vp=8.0,
        kernel_vs=4.5,
        kernel_ratio=4.5 / 8.0,
        rho_per_layer=rho_layers,
        pick_mae_p=float(pick_noise_std),
        pick_mae_s=float(pick_noise_std),
        meta={"synthetic_fallback": True, "raw_tp": obs_tp.cpu().tolist(), "raw_ts": obs_ts.cpu().tolist()},
    )


def prior_to_dict(prior: PickingPrior) -> dict[str, Any]:
    return {
        "vp_init": prior.vp_init.detach().cpu().tolist(),
        "vs_init": prior.vs_init.detach().cpu().tolist(),
        "q_init": prior.q_init.detach().cpu().tolist(),
        "obs_tp": prior.obs_tp.detach().cpu().tolist(),
        "obs_ts": prior.obs_ts.detach().cpu().tolist(),
        "kernel_vp": prior.kernel_vp,
        "kernel_vs": prior.kernel_vs,
        "kernel_ratio": prior.kernel_ratio,
        "rho_per_layer": prior.rho_per_layer.detach().cpu().tolist(),
        "pick_mae_p": prior.pick_mae_p,
        "pick_mae_s": prior.pick_mae_s,
        "meta": prior.meta,
    }


def prior_from_dict(data: dict[str, Any], device: torch.device | str = "cpu") -> PickingPrior:
    dev = torch.device(device)
    return PickingPrior(
        vp_init=torch.tensor(data["vp_init"], device=dev, dtype=torch.float32),
        vs_init=torch.tensor(data["vs_init"], device=dev, dtype=torch.float32),
        q_init=torch.tensor(data["q_init"], device=dev, dtype=torch.float32),
        obs_tp=torch.tensor(data["obs_tp"], device=dev, dtype=torch.float32),
        obs_ts=torch.tensor(data["obs_ts"], device=dev, dtype=torch.float32),
        kernel_vp=float(data["kernel_vp"]),
        kernel_vs=float(data["kernel_vs"]),
        kernel_ratio=float(data.get("kernel_ratio", data["kernel_vs"] / max(data["kernel_vp"], 1e-6))),
        rho_per_layer=torch.tensor(data["rho_per_layer"], device=dev, dtype=torch.float32),
        pick_mae_p=float(data["pick_mae_p"]),
        pick_mae_s=float(data["pick_mae_s"]),
        meta=data.get("meta", {}),
    )


def save_prior_cache(prior: PickingPrior, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior_to_dict(prior), indent=2))


def load_prior_cache(path: Path, device: torch.device | str = "cpu") -> PickingPrior:
    return prior_from_dict(json.loads(path.read_text()), device=device)


def load_picking_model_from_checkpoint(checkpoint: Path, device: torch.device, bypass: bool = True):
    """Thin wrapper around analyze_stead_picking.load_model."""
    from analyze_stead_picking import load_model

    return load_model(checkpoint, device, bypass_noise_cancel=bypass)
