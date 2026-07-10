#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare rho(t) against classical seismic attributes on STEAD events.

Attributes:
  - envelope (Hilbert)
  - STA/LTA
  - instantaneous frequency (approx via phase derivative)
  - short-window energy ratio

Also exports multi-scene case panels for a paper case library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from analyze_stead_picking import load_model
from hnf.picking_metrics import idx_to_sec
from hnf.stead_picking_dataset import STEADPickingDataset
from run_knowledge_mining import spearman_corr, bootstrap_ci, _normal_p_from_r


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rho(t) vs classical attributes")
    p.add_argument("--checkpoint", default="outputs/run20/20_wrongpeak_sharp/best.pt")
    p.add_argument("--output-dir", default="outputs/paper_rho_attributes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=800)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--n-cases", type=int, default=12)
    p.add_argument("--seed", type=int, default=3)
    return p.parse_args()


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    # x: (T,)
    spec = np.fft.rfft(x)
    h = np.zeros_like(spec)
    n = len(spec)
    if n:
        h[0] = 1
        if len(x) % 2 == 0 and n > 1:
            h[-1] = 1
        h[1:-1 if len(x) % 2 == 0 else None] = 2
    analytic = np.fft.irfft(spec * h, n=len(x))
    # For real implementation use phase of complex; approximate envelope via energy of x and 90deg shift
    # Better: construct complex analytic signal
    spec2 = np.fft.fft(x)
    h2 = np.zeros(len(x))
    if len(x) % 2 == 0:
        h2[0] = h2[len(x) // 2] = 1
        h2[1:len(x) // 2] = 2
    else:
        h2[0] = 1
        h2[1:(len(x) + 1) // 2] = 2
    analytic = np.fft.ifft(spec2 * h2)
    return np.abs(analytic).astype(np.float64)


def sta_lta(x: np.ndarray, sta: int = 20, lta: int = 100) -> np.ndarray:
    x2 = x.astype(np.float64) ** 2
    c = np.cumsum(x2)
    def win_mean(n):
        out = np.zeros_like(x2)
        for i in range(len(x2)):
            i0 = max(0, i - n + 1)
            out[i] = (c[i] - (c[i0 - 1] if i0 > 0 else 0.0)) / max(i - i0 + 1, 1)
        return out
    sta_e = win_mean(sta)
    lta_e = win_mean(lta).clip(min=1e-12)
    return sta_e / lta_e


def inst_freq(x: np.ndarray, dt: float = 60.0 / 799.0) -> np.ndarray:
    spec = np.fft.fft(x)
    h = np.zeros(len(x))
    if len(x) % 2 == 0:
        h[0] = h[len(x) // 2] = 1
        h[1:len(x) // 2] = 2
    else:
        h[0] = 1
        h[1:(len(x) + 1) // 2] = 2
    analytic = np.fft.ifft(spec * h)
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase, prepend=phase[0]) / (2 * np.pi * max(dt, 1e-6))
    return np.abs(freq)


def energy_ratio(x: np.ndarray, win: int = 40) -> np.ndarray:
    x2 = x.astype(np.float64) ** 2
    out = np.zeros_like(x2)
    for i in range(len(x2)):
        i0 = max(0, i - win + 1)
        out[i] = x2[i0:i + 1].mean()
    return out


def peak_near(arr: np.ndarray, center: int, left=30, right=50) -> int:
    i0 = max(0, center - left)
    i1 = min(len(arr), center + right)
    return int(np.argmax(arr[i0:i1]) + i0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "cases"
    case_dir.mkdir(exist_ok=True)
    docs = Path("docs/figures")
    docs.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(Path(args.checkpoint), device, bypass_noise_cancel=True)
    ds = STEADPickingDataset("test", seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows = []
    cases = []
    n = 0
    for batch in loader:
        if n >= args.max_events:
            break
        if float(batch["det"][0]) <= 0.5 or float(batch["p_valid"][0]) <= 0 or float(batch["s_valid"][0]) <= 0:
            continue
        dist = float(batch["source_distance_km"][0])
        depth = float(batch["source_depth_km"][0])
        if not np.isfinite(dist) or dist < 1 or dist > 200:
            continue
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        with torch.no_grad():
            out = model.forward_explain(x, t)
        rho = out["rho"][0].detach().cpu().numpy()
        z = x[0, :, 2].detach().cpu().numpy()  # Z component
        env = hilbert_envelope(z)
        sl = sta_lta(z)
        ifr = inst_freq(z)
        eng = energy_ratio(z)
        p_idx = int(batch["p_idx"][0])
        s_idx = int(batch["s_idx"][0])
        gt_p = idx_to_sec(p_idx, x.shape[1])
        gt_s = idx_to_sec(s_idx, x.shape[1])
        t_sec = t[0, :, 0].detach().cpu().numpy()

        # normalize for correlation in windows around P/S
        def win_corr(a, b, center):
            i0 = max(0, center - 40)
            i1 = min(len(a), center + 60)
            aa = a[i0:i1]
            bb = b[i0:i1]
            if aa.std() < 1e-12 or bb.std() < 1e-12:
                return float("nan")
            return float(np.corrcoef(aa, bb)[0, 1])

        row = {
            "trace_name": str(batch["trace_name"][0]),
            "distance_km": dist,
            "source_depth_km": max(depth, 1.0),
            "rho_env_corr_p": win_corr(rho, env, p_idx),
            "rho_stalta_corr_p": win_corr(rho, sl, p_idx),
            "rho_energy_corr_p": win_corr(rho, eng, p_idx),
            "rho_env_corr_s": win_corr(rho, env, s_idx),
            "rho_stalta_corr_s": win_corr(rho, sl, s_idx),
            "rho_p_lag": float(t_sec[peak_near(rho, p_idx)] - gt_p),
            "env_p_lag": float(t_sec[peak_near(env, p_idx)] - gt_p),
            "stalta_p_lag": float(t_sec[peak_near(sl, p_idx)] - gt_p),
            "rho_s_lag": float(t_sec[peak_near(rho, s_idx)] - gt_s),
            "env_s_lag": float(t_sec[peak_near(env, s_idx)] - gt_s),
            "stalta_s_lag": float(t_sec[peak_near(sl, s_idx)] - gt_s),
        }
        rows.append(row)

        # scene tags for case library
        scene = "mid"
        if dist < 40:
            scene = "near"
        elif dist > 120:
            scene = "far"
        if depth > 20:
            scene = scene + "_deep"
        elif depth < 8:
            scene = scene + "_shallow"

        if len(cases) < args.n_cases:
            cases.append({
                "trace": row["trace_name"],
                "scene": scene,
                "distance_km": dist,
                "source_depth_km": max(depth, 1.0),
                "z": z,
                "rho": rho,
                "env": env,
                "stalta": sl,
                "t_sec": t_sec,
                "gt_p": gt_p,
                "gt_s": gt_s,
            })
        n += 1
        if n % 50 == 0:
            print(f"[attrs] {n}/{args.max_events}", flush=True)

    # summary stats
    summary = {}
    for key in [
        "rho_env_corr_p", "rho_stalta_corr_p", "rho_energy_corr_p",
        "rho_env_corr_s", "rho_stalta_corr_s",
    ]:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        summary[key] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }
    lag_pairs = [
        ("rho_p_lag", "env_p_lag"),
        ("rho_p_lag", "stalta_p_lag"),
        ("rho_s_lag", "env_s_lag"),
        ("rho_s_lag", "stalta_s_lag"),
    ]
    lag_stats = []
    for a, b in lag_pairs:
        xa = [r[a] for r in rows]
        yb = [r[b] for r in rows]
        sp = spearman_corr(xa, yb)
        ci = bootstrap_ci(xa, yb, n_boot=200, seed=args.seed)
        lag_stats.append({
            "x": a, "y": b, "spearman": sp, "ci95": list(ci),
            "p_approx": _normal_p_from_r(sp, len(rows)),
        })

    # overview figure
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    labels = ["env@P", "STA/LTA@P", "energy@P", "env@S", "STA/LTA@S"]
    keys = ["rho_env_corr_p", "rho_stalta_corr_p", "rho_energy_corr_p", "rho_env_corr_s", "rho_stalta_corr_s"]
    means = [summary[k]["mean"] for k in keys]
    stds = [summary[k]["std"] for k in keys]
    axes[0].bar(range(len(labels)), means, yerr=stds, color="C0", alpha=0.85, capsize=3)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("window Pearson corr with rho(t)")
    axes[0].set_title(f"rho(t) vs classical attributes (n={len(rows)})")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].scatter([r["rho_p_lag"] for r in rows], [r["env_p_lag"] for r in rows], s=18, alpha=0.7, label="env")
    axes[1].scatter([r["rho_p_lag"] for r in rows], [r["stalta_p_lag"] for r in rows], s=18, alpha=0.7, label="STA/LTA")
    axes[1].plot([-2, 2], [-2, 2], "k--", lw=1)
    axes[1].set_xlabel("rho peak lag @P (s)")
    axes[1].set_ylabel("attribute peak lag @P (s)")
    axes[1].set_title("Peak-lag agreement around P")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    p_sum = out_dir / "rho_vs_attributes_summary.png"
    fig.savefig(p_sum, dpi=160)
    plt.close(fig)
    (docs / "rho_vs_attributes_summary.png").write_bytes(p_sum.read_bytes())

    # case library panels
    for i, c in enumerate(cases):
        fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
        axes[0].plot(c["t_sec"], c["z"], color="0.2", lw=0.8)
        axes[0].set_ylabel("Z")
        axes[0].set_title(f"{c['trace']} | {c['scene']} | d={c['distance_km']:.1f}km z={c['source_depth_km']:.1f}km")
        axes[1].plot(c["t_sec"], c["rho"], color="C0")
        axes[1].set_ylabel("rho(t)")
        axes[2].plot(c["t_sec"], c["env"] / (np.max(c["env"]) + 1e-8), color="C1", label="env")
        axes[2].plot(c["t_sec"], c["stalta"] / (np.max(c["stalta"]) + 1e-8), color="C2", label="STA/LTA")
        axes[2].legend(fontsize=8, loc="upper right")
        axes[2].set_ylabel("attr (norm)")
        axes[3].axis("off")
        axes[3].text(0.01, 0.6, "vertical lines: GT P (red), GT S (orange)", fontsize=9)
        for ax in axes[:3]:
            ax.axvline(c["gt_p"], color="red", ls="--", lw=1)
            ax.axvline(c["gt_s"], color="orange", ls="--", lw=1)
            ax.grid(True, alpha=0.25)
        axes[2].set_xlabel("time (s)")
        fp = case_dir / f"case_{i:02d}_{c['scene']}.png"
        fig.savefig(fp, dpi=140)
        plt.close(fig)

    report = {
        "n_events": len(rows),
        "summary": summary,
        "lag_stats": lag_stats,
        "n_cases": len(cases),
        "case_dir": str(case_dir),
        "figure": str(p_sum),
    }
    (out_dir / "rho_attributes_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# rho(t) vs Classical Attributes",
        "",
        f"- n events: {len(rows)}",
        f"- cases: {len(cases)} in `{case_dir.name}/`",
        "",
        "## Window correlations",
    ]
    for k in keys:
        s = summary[k]
        md.append(f"- `{k}`: mean={s['mean']:.3f} ± {s['std']:.3f} (median={s['median']:.3f})")
    md += ["", "## Peak-lag Spearman"]
    for r in lag_stats:
        md.append(f"- `{r['x']}` vs `{r['y']}`: {r['spearman']:.3f}, CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]")
    (out_dir / "rho_attributes_report.md").write_text("\n".join(md))
    print(json.dumps({"n": len(rows), "figure": str(p_sum), "report": str(out_dir / "rho_attributes_report.json")}, indent=2))


if __name__ == "__main__":
    main()
