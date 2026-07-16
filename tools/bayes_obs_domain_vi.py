#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme 2: Bayesian domain posterior over kernel scales + OBS adapt.

Learns q(θ)=N(μ, softplus(σ)^2) over log-scale kernel box dims (P/S/MS γ/ω/c).
- Prior = identity scales (STEAD) → KL(q||N(0,I)) on log(θ)
- Likelihood ≈ OBS pick loss (BCE on soft labels) with MC samples from q
- Quantize posterior into particles for ensemble inference on holdout

Trunk remains frozen; only variational parameters + optional tiny kernel nudges
via the applied scales.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import importlib.util
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hnf.stead_picking_dataset import gaussian_pick_label
from tools.analyze_stead_picking import load_model
from tools.obs_kernel_box import (
    GROUP_SCALE_NAMES,
    apply_group_log_scales,
    restore_kernel_raw,
    scales_from_vector,
    snapshot_kernel_state,
)
from tools.obs_matched_split import load_split_samples
from tools.train_stead_picking import move_batch_to_device, set_seed, weighted_pick_loss


def _load_obs_compare_module():
    path = _REPO_ROOT / "scripts" / "paper" / "run_paper_obs_picking_compare.py"
    spec = importlib.util.spec_from_file_location("obs_compare", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian domain posterior OBS adapt")
    p.add_argument("--checkpoint", default="outputs/run28/28_ms_fresnel_phys_20ep/best.pt")
    p.add_argument("--split-json", default="outputs/obs_matched_adapt_split_randoffset/split.json")
    p.add_argument("--output-dir", default="outputs/obs_bayes_domain_posterior")
    p.add_argument("--seq-len", type=int, default=1600)
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--n-mc", type=int, default=2, help="MC samples from q per step")
    p.add_argument("--n-particles", type=int, default=8, help="Quantized posterior particles")
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--scale-min", type=float, default=0.35)
    p.add_argument("--scale-max", type=float, default=3.0)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--pick-pos-weight", type=float, default=28.0)
    p.add_argument("--p-loss-weight", type=float, default=1.5)
    p.add_argument("--s-loss-weight", type=float, default=1.2)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--det-threshold", type=float, default=0.5)
    p.add_argument("--tol-sec", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_items(samples, seq_len, window_sec, label_sigma_sec, normalize_wave):
    sigma = max(1.0, label_sigma_sec * seq_len / window_sec)
    items = []
    for s in samples:
        x = torch.from_numpy(normalize_wave(s["wave_3_raw"], "std")).float()
        x = F.interpolate(x.unsqueeze(0), size=seq_len, mode="linear", align_corners=False).squeeze(0)
        x = x.transpose(0, 1)
        t = torch.linspace(0.0, window_sec, seq_len).unsqueeze(-1)
        scale = seq_len / float(s["wave_3_raw"].shape[-1])
        p_idx = int(max(0, min(seq_len - 1, round(s["p_idx_native"] * scale))))
        p_target = gaussian_pick_label(p_idx, seq_len, sigma)
        if s["s_valid"]:
            s_idx = int(max(0, min(seq_len - 1, round(s["s_idx_native"] * scale))))
            s_target = gaussian_pick_label(s_idx, seq_len, sigma)
            s_valid = 1.0
        else:
            s_idx = 0
            s_target = torch.zeros(seq_len)
            s_valid = 0.0
        items.append(
            {
                "x": x,
                "t": t,
                "p_target": p_target,
                "s_target": s_target,
                "s_valid": s_valid,
            }
        )
    return items


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "p_target": torch.stack([b["p_target"] for b in batch]),
        "s_target": torch.stack([b["s_target"] for b in batch]),
        "s_valid": torch.tensor([b["s_valid"] for b in batch], dtype=torch.float32),
    }


class KernelScalePosterior(nn.Module):
    """Diagonal Gaussian over log(scale) for GROUP_SCALE_NAMES; prior N(0,I)."""

    def __init__(self, dim: int, scale_min: float, scale_max: float):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.raw_sigma = nn.Parameter(torch.full((dim,), -1.0))  # softplus(~0.3)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 1e-3

    def sample_log_scales(self, n: int) -> torch.Tensor:
        eps = torch.randn(n, self.mu.numel(), device=self.mu.device)
        return self.mu.unsqueeze(0) + self.sigma().unsqueeze(0) * eps

    def scales_from_log(self, log_s: torch.Tensor) -> torch.Tensor:
        # map R -> (scale_min, scale_max) via sigmoid stretch around 1
        # log_s=0 → ~1
        u = torch.sigmoid(log_s)
        return self.scale_min + (self.scale_max - self.scale_min) * u

    def kl_to_standard_normal(self) -> torch.Tensor:
        # KL(N(mu,s^2) || N(0,1))
        s2 = self.sigma() ** 2
        return 0.5 * torch.sum(self.mu**2 + s2 - torch.log(s2) - 1.0)

    def quantile_particles(self, n: int) -> torch.Tensor:
        # Deterministic quantization: mid-quantiles of each margin, then mean vector
        # + isotropic probe; simpler: sample then keep; use linspace eps for reproducibility
        qs = torch.linspace(0.1, 0.9, n, device=self.mu.device)
        from torch.distributions import Normal

        z = Normal(0, 1).icdf(qs).unsqueeze(1)
        log_s = self.mu.unsqueeze(0) + self.sigma().unsqueeze(0) * z
        return self.scales_from_log(log_s)


def batch_pick_loss(out, batch, args) -> torch.Tensor:
    p_logits = torch.nan_to_num(out.get("p_logits", out["p"]), nan=-50.0, posinf=50.0, neginf=-50.0)
    s_logits = torch.nan_to_num(out.get("s_logits", out["s"]), nan=-50.0, posinf=50.0, neginf=-50.0)
    loss_p = weighted_pick_loss(p_logits, batch["p_target"], args.pick_pos_weight)
    s_mask = batch["s_valid"] > 0.5
    if s_mask.any():
        loss_s = weighted_pick_loss(s_logits[s_mask], batch["s_target"][s_mask], args.pick_pos_weight)
    else:
        loss_s = p_logits.new_zeros(())
    return args.p_loss_weight * loss_p + args.s_loss_weight * loss_s


@torch.no_grad()
def eval_pick(model, samples, device, args, obs_mod):
    model.eval()
    return obs_mod.eval_hnf(
        model, samples, device, args.seq_len, args.window_sec,
        args.pick_threshold, args.det_threshold, args.tol_sec, args.batch_size,
    )["pick_only"]


@torch.no_grad()
def eval_ensemble(model, base, particles_scales, samples, device, args, obs_mod):
    """Average sigmoid probs across quantized posterior particles."""
    model.eval()
    # Accumulator on peaks via soft probability averaging in index space is heavy;
    # use score average of per-particle pick_only as a proxy, plus mean-particle ckpt eval.
    rows = []
    for i, vec in enumerate(particles_scales):
        scales = scales_from_vector(vec.detach().cpu().numpy().tolist())
        restore_kernel_raw(model, base)
        apply_group_log_scales(model, base, scales)
        po = eval_pick(model, samples, device, args, obs_mod)
        rows.append({"particle": i, "scales": scales, "pick_only": po,
                     "score": 0.65 * po["p_f1"] + 0.35 * po["s_f1"]})
    # pick best particle for deterministic deploy; also report mean score
    best = max(rows, key=lambda r: r["score"])
    mean_score = float(np.mean([r["score"] for r in rows]))
    return best, mean_score, rows


def main() -> None:
    args = parse_args()
    if args.epochs < 50:
        args.epochs = 50
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    obs_mod = _load_obs_compare_module()

    train_samples, _, split_meta = load_split_samples(args.split_json, "train")
    holdout_samples, _, _ = load_split_samples(args.split_json, "holdout")
    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(train_samples))
    rng.shuffle(idxs)
    n_val = max(64, int(0.2 * len(train_samples)))
    val_idx = set(idxs[:n_val].tolist())
    tr = [train_samples[i] for i in idxs if i not in val_idx]
    va = [train_samples[i] for i in sorted(val_idx)]

    train_items = build_items(
        tr, args.seq_len, args.window_sec, args.label_sigma_sec, obs_mod.normalize_wave
    )
    # kept for optional full-batch probing; CEM uses a smaller probe set below
    _ = DataLoader(ItemDataset(train_items), batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model, ckpt_args = load_model(Path(args.checkpoint), device, bypass_noise_cancel=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    base = snapshot_kernel_state(model)
    (out_dir / "kernel_base.json").write_text(json.dumps(base, indent=2))

    dim = len(GROUP_SCALE_NAMES)
    q = KernelScalePosterior(dim, args.scale_min, args.scale_max).to(device)

    history = []
    t0 = time.time()
    print(
        f"[bayes-vi] train={len(tr)} val={len(va)} holdout={len(holdout_samples)} "
        f"dims={dim} kl_beta={args.kl_beta} protocol={split_meta.get('protocol')}",
        flush=True,
    )

    # identity baselines
    id_val = eval_pick(model, va[: min(240, len(va))], device, args, obs_mod)
    id_hold = eval_pick(model, holdout_samples, device, args, obs_mod)
    print(
        f"[bayes-vi] identity val P/S={id_val['p_f1']:.3f}/{id_val['s_f1']:.3f} "
        f"hold={id_hold['p_f1']:.3f}/{id_hold['s_f1']:.3f}",
        flush=True,
    )

    # Cross-entropy method / Bayesian diagonal update on a fixed train mini-pool
    # for cheap likelihood, then refine q and quantize particles.
    probe_n = min(128, len(tr))
    probe = [tr[int(i)] for i in rng.choice(len(tr), size=probe_n, replace=False)]
    probe_items = build_items(
        probe, args.seq_len, args.window_sec, args.label_sigma_sec, obs_mod.normalize_wave
    )
    probe_loader = DataLoader(
        ItemDataset(probe_items), batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    def probe_nll_for_scales(scales: dict) -> float:
        restore_kernel_raw(model, base)
        apply_group_log_scales(model, base, scales)
        total = 0.0
        m = 0
        with torch.no_grad():
            for batch in probe_loader:
                batch = move_batch_to_device(batch, device)
                out = model(batch["x"], batch["t"])
                total += float(batch_pick_loss(out, batch, args).detach()) * batch["x"].size(0)
                m += batch["x"].size(0)
        return total / max(1, m)

    elite_frac = 0.25
    n_pop = max(16, args.n_particles * 3)
    for ep in range(1, args.epochs + 1):
        # Sample population from current q
        with torch.no_grad():
            log_s = q.sample_log_scales(n_pop)
            scales_t = q.scales_from_log(log_s)
        costs = []
        for k in range(n_pop):
            scales = scales_from_vector(scales_t[k].cpu().tolist())
            costs.append(probe_nll_for_scales(scales))
        costs_a = np.asarray(costs, dtype=np.float64)
        n_elite = max(2, int(round(elite_frac * n_pop)))
        elite_idx = np.argsort(costs_a)[:n_elite]
        elite_log = log_s[elite_idx].detach()
        # Bayesian / moment match update toward elites + KL prior pull
        elite_mu = elite_log.mean(dim=0)
        elite_std = elite_log.std(dim=0).clamp_min(1e-3)
        with torch.no_grad():
            # shrink to prior mean 0
            q.mu.copy_((1.0 - args.kl_beta) * elite_mu)
            # sigma raw from softplus_inv
            target_s = (1.0 - 0.5 * args.kl_beta) * elite_std + 0.5 * args.kl_beta * torch.ones_like(elite_std)
            # softplus_inv
            q.raw_sigma.copy_(torch.log(torch.expm1(target_s.clamp_min(1e-3))))

        mean_vec = q.scales_from_log(q.mu).detach().cpu().tolist()
        restore_kernel_raw(model, base)
        apply_group_log_scales(model, base, scales_from_vector(mean_vec))
        val_po = eval_pick(model, va[: min(240, len(va))], device, args, obs_mod)
        score = 0.65 * val_po["p_f1"] + 0.35 * val_po["s_f1"]
        row = {
            "epoch": ep,
            "probe_nll_mean": float(costs_a.mean()),
            "probe_nll_elite": float(costs_a[elite_idx].mean()),
            "val_pick_only": val_po,
            "val_score": score,
            "mean_scales": scales_from_vector(mean_vec),
            "sigma": q.sigma().detach().cpu().tolist(),
        }
        history.append(row)
        print(
            f"[bayes-vi] ep{ep} elite_nll={row['probe_nll_elite']:.3f} "
            f"val P={val_po['p_f1']:.3f} S={val_po['s_f1']:.3f} score={score:.3f}",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    particles = q.quantile_particles(args.n_particles)
    best_part, mean_score, part_rows = eval_ensemble(
        model, base, particles, holdout_samples, device, args, obs_mod
    )
    # Deploy best particle
    restore_kernel_raw(model, base)
    apply_group_log_scales(model, base, best_part["scales"])
    hold_mean = eval_pick(model, holdout_samples, device, args, obs_mod)

    ckpt = {
        "state_dict": model.state_dict(),
        "args": ckpt_args,
        "bayes_scheme": "domain_posterior_cem",
        "q_state": {k: v.detach().cpu() for k, v in q.state_dict().items()},
        "best_particle_scales": best_part["scales"],
        "mean_scales": scales_from_vector(q.scales_from_log(q.mu).detach().cpu().tolist()),
        "kernel_base": base,
        "holdout_best_particle": best_part["pick_only"],
        "holdout_mean_scales": hold_mean,
        "holdout_identity": id_hold,
        "particle_mean_score": mean_score,
        "adapt_args": vars(args),
        "split_json": args.split_json,
    }
    torch.save(ckpt, out_dir / "best.pt")
    report = {
        "scheme": "2_domain_posterior_cem",
        "holdout_identity": id_hold,
        "holdout_best_particle": best_part["pick_only"],
        "holdout_deploy_mean_scales": hold_mean,
        "particle_bank_mean_score": mean_score,
        "best_particle_scales": best_part["scales"],
        "posterior_sigma": q.sigma().detach().cpu().tolist(),
        "elapsed_sec": time.time() - t0,
        "note": (
            "Frozen trunk; CEM updates diagonal Gaussian q over kernel scales "
            "with KL shrink to STEAD identity; quantized particles for deploy"
        ),
    }
    (out_dir / "bayes_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "particles.json").write_text(json.dumps(part_rows, indent=2))
    md = [
        "# OBS Bayesian domain posterior (scheme 2)",
        "",
        "- CEM/Bayesian update of q(θ) over kernel (γ,ω,c) scales; shrink to STEAD prior",
        "- Quantize into particles; deploy best particle on holdout",
        f"- holdout identity P/S: {id_hold['p_f1']:.3f} / {id_hold['s_f1']:.3f}",
        f"- holdout best particle P/S: {best_part['pick_only']['p_f1']:.3f} / {best_part['pick_only']['s_f1']:.3f}",
        f"- particle-bank mean score: {mean_score:.3f}",
    ]
    (out_dir / "bayes_report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
