#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned Huygens kernel parameters per layer, in physical units (CPU only).

Per block: wave speed c, damping γ [s^-2] with envelope e-fold 1/sqrt(γ) [s], ω [rad/s] with period 2π/ω,
obliquity α, and effective memory = lags holding 50% / 90% of Σ|K(lag)| inside the light cone.
Also plots |K(lag)| and Re K(lag) for each branch.

  python tools/analyze_kernel_params.py --c2f final=outputs/stead_c2f_lft_RSP/best.pt,rob=outputs/stead_c2f_rob_RSP/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402

BRANCHES = ("shared", "p_layers", "s_layers")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c2f", required=True, help="name=ckpt,...")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paper_stead_final_compare/final/interp"))
    return p.parse_args()


@torch.no_grad()
def block_profile(k, coarse_len):
    dt = 60.0 / max(coarse_len - 1, 1)
    w = k.window_bins(coarse_len)
    lags = torch.arange(1, w + 1, dtype=torch.float32).view(w, 1, 1) * dt
    coef = k._band_lag_coeffs(lags)
    mag = coef.abs().float()
    cum = torch.cumsum(mag, 0) / mag.sum().clamp_min(1e-12)
    lag = lags.view(-1)
    g, om, c = float(k.effective_gamma()), float(k.effective_omega()), float(k.effective_wave_speed())
    return {
        "c": c,
        "gamma": g,
        "envelope_efold_s": 1.0 / math.sqrt(max(g, 1e-9)),
        "omega": om,
        "period_s": 2 * math.pi / max(om, 1e-9),
        "obliquity_alpha": float(k.effective_obliquity_scale()),
        "window_s": float(k.local_window_sec or 60.0),
        "mem50_s": float(lag[int((cum < 0.5).sum())]),
        "mem90_s": float(lag[min(int((cum < 0.9).sum()), w - 1)]),
    }, lag.numpy(), coef


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = dict(s.split("=") for s in args.c2f.split(",") if s)
    report = {}
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 3.6 * len(models)), squeeze=False)
    for row, (name, path) in enumerate(models.items()):
        model = load_model(Path(path), torch.device("cpu"))[0].eval()
        rep = {}
        for col, br in enumerate(BRANCHES):
            rep[br] = []
            ax = axes[row, col]
            for i, blk in enumerate(getattr(model, br)):
                prof, lag, coef = block_profile(blk.kernel, model.coarse_len)
                rep[br].append(prof)
                m = coef.abs().numpy()
                ax.plot(lag, m / m.max(), label=f"L{i} |K| (c={prof['c']:.2f}, γ={prof['gamma']:.3f}, ω={prof['omega']:.2f})")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.5)
            ax.set_title(f"{name}: {br}")
            ax.set_xlabel("lag [s]")
            ax.legend(fontsize=7)
        cp = sum(r["c"] for r in rep["p_layers"]) / len(rep["p_layers"])
        cs = sum(r["c"] for r in rep["s_layers"]) / len(rep["s_layers"])
        rep["vp_over_vs"] = cp / max(cs, 1e-6)
        rep["mem90_p_mean"] = sum(r["mem90_s"] for r in rep["p_layers"]) / len(rep["p_layers"])
        rep["mem90_s_mean"] = sum(r["mem90_s"] for r in rep["s_layers"]) / len(rep["s_layers"])
        report[name] = rep
        print(f"== {name}  vp/vs={rep['vp_over_vs']:.2f}  mem90 P={rep['mem90_p_mean']:.2f}s S={rep['mem90_s_mean']:.2f}s")
        for br in BRANCHES:
            for i, r in enumerate(rep[br]):
                print(f"  {br:9s} L{i} c={r['c']:.3f} γ={r['gamma']:.4f} (e-fold {r['envelope_efold_s']:.1f}s) "
                      f"ω={r['omega']:.3f} (T {r['period_s']:.1f}s) α={r['obliquity_alpha']:.2f} "
                      f"mem50={r['mem50_s']:.2f}s mem90={r['mem90_s']:.2f}s")
    fig.tight_layout()
    fig.savefig(args.output_dir / "kernel_profiles.png", dpi=130)
    (args.output_dir / "kernel_params.json").write_text(json.dumps(report, indent=2))
    print(f"[done] → {args.output_dir}")


if __name__ == "__main__":
    main()
