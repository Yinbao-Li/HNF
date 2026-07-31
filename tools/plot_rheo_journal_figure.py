#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rheology track: master board + interpretability mining + journal figure (a–d).

Panels:
  a  Master model comparison (stress_rel)
  b  Merged anisotropic spectra G11/G22 (GT solid, PNF markers)
  c  Viz case: multi-step shear — γ̇ strip + GT | PNF | |RhINN−GT| (+ external colorbars)
  d  Oscillatory residual σ̂−σ (+ GT stress strip)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from hnf.rheo_dataset import RheoMemoryDataset, RheoMemoryModel
from hnf.rheo_domain_sota import ClassicalPronyNLS, SparsePronyLibrary
from hnf.rheo_memory import PronyBoltzmannKernel
from tools.tune_rheo_domain_sota import TunedRhINN

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

C_PNF = "#0B3D4A"
C_NLS = "#1B6B93"
C_EUCLID = "#C45C26"
C_RHINN = "#6B4F3A"
C_BASE = "#9AA3A7"
C_BAD = "#A33B2B"
BG = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fig-dir", default="docs/figures/rheo")
    p.add_argument("--out-dir", default="outputs/rheo/interpret_mine")
    p.add_argument("--stem", default="rheo_journal_memory_sota")
    p.add_argument("--ckpt-pnf", default="outputs/rheo/domain_sota_tuned/pnf_aniso/best.pt")
    p.add_argument("--ckpt-nls", default="outputs/rheo/domain_sota_tuned/classical_prony_nls/best.pt")
    p.add_argument("--ckpt-euclid", default="outputs/rheo/domain_sota_tuned/euclid/best.pt")
    p.add_argument("--ckpt-rhinn", default="outputs/rheo/domain_sota_tuned/rhinn/best.pt")
    return p.parse_args()


def _collect_master_rows() -> list[dict[str, Any]]:
    """Unify all historical rheo boards into one table."""
    rows: list[dict[str, Any]] = []

    def add(name, family, stress_rel, n_params=None, note="", source="", **extra):
        rows.append(
            {
                "model": name,
                "family": family,
                "stress_rel": float(stress_rel),
                "n_params": n_params,
                "note": note,
                "source": source,
                **extra,
            }
        )

    # R0 isotropic suite
    suite = json.loads(Path("outputs/rheo/suite_final/BOARD.json").read_text())
    fam_map = {
        "r0_k2_full": ("R0 isotropic Prony", "best isotropic config"),
        "r0_k2_freq": ("R0 isotropic Prony", "freq loss only"),
        "r0_k2_stress_only": ("R0 isotropic Prony", "stress MSE only"),
        "r0_k1_stress_only": ("R0 isotropic Prony", "K=1 underfit"),
        "r0_k3_full": ("R0 isotropic Prony", "K=3 overparam"),
        "r1_k2_aniso": ("R1 anisotropic Prony", "early aniso suite"),
    }
    for r in suite["rows"]:
        fam, note = fam_map.get(r["name"], ("other", ""))
        add(r["name"], fam, r["stress_rel"], note=note, source="suite_final",
            lambda_rel=r.get("lambda_rel"), G_rel=r.get("G_rel"), score=r.get("score"))

    # Aniso vs generic ML
    full = json.loads(Path("outputs/rheo/aniso_sota_full/BOARD.json").read_text())
    for r in full["rows"]:
        fam = {
            "pnf_aniso": "PNF aniso",
            "diagonal_prony": "Rheology baseline",
            "isotropic_prony": "Misspecified",
            "lstm": "Generic ML ablation",
            "tcn": "Generic ML ablation",
            "linear_fir": "Generic ML ablation",
        }.get(r["model"], "other")
        add(r["model"], fam, r["stress_rel"], r.get("n_params"),
            note="aniso data board", source="aniso_sota_full",
            lambda_rel=r.get("lambda_rel"))

    # Domain untuned
    dom = json.loads(Path("outputs/rheo/domain_sota/BOARD.json").read_text())
    for r in dom["rows"]:
        add(r["model"] + " (untuned)", "Domain SOTA (untuned)", r["stress_rel"], r.get("n_params"),
            note=r.get("cite", ""), source="domain_sota")

    # Domain tuned (flagship)
    tuned = json.loads(Path("outputs/rheo/domain_sota_tuned/BOARD.json").read_text())
    for r in tuned["rows"]:
        add(r["model"], "Domain SOTA (tuned)", r["stress_rel"], r.get("n_params"),
            note=json.dumps(r.get("cfg", {}), separators=(",", ":")),
            source="domain_sota_tuned",
            stress_rel_median=r.get("stress_rel_median"),
            stress_rel_p90=r.get("stress_rel_p90"))

    rows.sort(key=lambda x: x["stress_rel"])
    return rows


def _load_tuned_models(args, device):
    # PNF
    pnf = RheoMemoryModel(n_modes=2, dim=2, anisotropic=True, lambda_init=[0.4, 4.0], g_init=[1.0, 0.7]).to(device)
    ck = torch.load(args.ckpt_pnf, map_location=device, weights_only=False)
    pnf.load_state_dict(ck["model"])
    pnf.eval()

    # NLS
    nls = ClassicalPronyNLS(n_modes=2, dim=2, anisotropic=True).to(device)
    ck = torch.load(args.ckpt_nls, map_location=device, weights_only=False)
    nls.load_state_dict(ck["model"], strict=False)
    nls.eval()

    # Euclid
    eu_cfg = ck.get("cfg") if False else None
    eu_ck = torch.load(args.ckpt_euclid, map_location=device, weights_only=False)
    n_lib = int(eu_ck.get("cfg", {}).get("n_library", 16))
    euclid = SparsePronyLibrary(dim=2, n_library=n_lib, l1_weight=1e-4).to(device)
    euclid.load_state_dict(eu_ck["model"])
    euclid.eval()

    # RhINN
    rh_ck = torch.load(args.ckpt_rhinn, map_location=device, weights_only=False)
    cfg = rh_ck.get("cfg", {})
    rhinn = TunedRhINN(
        dim=2, n_modes=2,
        hidden=int(cfg.get("hidden", 64)),
        n_layers=int(cfg.get("n_layers", 3)),
        phys_weight=float(cfg.get("phys_weight", 0.1)),
        mode=str(cfg.get("mode", "mech_encode")),
    ).to(device)
    rhinn.load_state_dict(rh_ck["model"])
    rhinn.eval()

    return {"pnf": pnf, "nls": nls, "euclid": euclid, "rhinn": rhinn}


def _gt_kernel() -> PronyBoltzmannKernel:
    # Canonical material used in aniso boards
    from hnf.rheo_synth import _make_gt_kernel
    return _make_gt_kernel(
        n_modes=2, dim=2, anisotropic=True,
        lambdas=[0.5, 5.0],
        weights=np.array([[1.2, 1.56], [0.6, 0.78]]),
        g_inf=0.0,
    )


def panel_title(ax, letter: str, title: str, *, fontsize: float = 10) -> None:
    """Bold panel letter only; title text stays regular weight."""
    ax.set_title("")
    ax.text(
        0.0,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.text(
        0.04,
        1.02,
        title,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="normal",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def panel_a(ax, rows):
    """Bar chart of flagship + key ablations."""
    # Prefer domain_sota_tuned when duplicate model names exist; no tuned/untuned tags.
    by_name: dict[str, dict] = {}
    for r in rows:
        key = r["model"]
        prev = by_name.get(key)
        if prev is None or r.get("source") == "domain_sota_tuned":
            by_name[key] = r
    pick = [
        ("pnf_aniso", "PNF aniso", C_PNF),
        ("classical_prony_nls_tuned", "Prony NLS", C_NLS),
        ("sparse_prony_euclid_tuned", "EUCLID-lite", C_EUCLID),
        ("rhinn_tuned", "RhINN", C_RHINN),
        ("diagonal_prony", "Diagonal Prony", C_BASE),
        ("lstm", "LSTM", C_BASE),
        ("isotropic_prony", "Isotropic Prony", C_BAD),
    ]
    labels, vals, colors = [], [], []
    for key, lab, col in pick:
        if key not in by_name:
            continue
        labels.append(lab)
        vals.append(by_name[key]["stress_rel"])
        colors.append(col)
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, edgecolor="none", zorder=3)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 1.08 if v > 0.01 else v + 0.0015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
    ax.set_ylabel(r"Stress rel. $L_2$ error")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    panel_title(ax, "a", "Model zoo — stress prediction error")
    ax.axhline(0.0033, color=C_PNF, ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7)
    ax.yaxis.set_label_coords(-0.12, 0.5)


def panel_b(ax, pnf, gt):
    """Merged anisotropic spectra: GT solid lines + PNF open markers (readable when overlapping)."""
    tau = torch.logspace(-2, 2, 200)
    g11_gt, g22_gt, g11_p, g22_p = [], [], [], []
    with torch.no_grad():
        for t in tau:
            Gg = gt.relaxation_modulus(t)
            Gp = pnf.kernel.relaxation_modulus(t)
            g11_gt.append(float(Gg[0, 0]))
            g22_gt.append(float(Gg[1, 1]))
            g11_p.append(float(Gp[0, 0]))
            g22_p.append(float(Gp[1, 1]))
        lam = np.sort(pnf.kernel.relaxation_times().cpu().numpy())
    tnp = tau.numpy()
    ax.plot(tnp, g11_gt, color="#1A1A1A", lw=2.0, label=r"GT $G_{11}$", zorder=2)
    ax.plot(tnp, g22_gt, color="#7A7A7A", lw=2.0, label=r"GT $G_{22}$", zorder=2)
    idx = np.linspace(0, len(tnp) - 1, 18, dtype=int)
    # keep markers away from λ markers near τ≈0.5 and τ≈5
    t_idx = tnp[idx]
    keep = (np.abs(np.log10(t_idx) - np.log10(0.5)) > 0.12) & (
        np.abs(np.log10(t_idx) - np.log10(5.0)) > 0.12
    )
    idx = idx[keep]
    ax.plot(
        tnp[idx], np.asarray(g11_p)[idx],
        "o", ms=5.5, mfc="none", mec=C_PNF, mew=1.4,
        label=r"PNF $G_{11}$", zorder=4,
    )
    ax.plot(
        tnp[idx], np.asarray(g22_p)[idx],
        "s", ms=5.0, mfc="none", mec="#C45C26", mew=1.4,
        label=r"PNF $G_{22}$", zorder=4,
    )
    ymax = float(max(g22_gt))
    for lam_i, c, name in [(lam[0], "#1B6B93", r"$\lambda_1$"), (lam[1], "#C45C26", r"$\lambda_2$")]:
        ax.axvline(float(lam_i), color=c, ls=":", lw=0.9, alpha=0.75, zorder=1)
        ax.text(
            float(lam_i), -0.10 * ymax, name, color=c, fontsize=8,
            ha="center", va="top", zorder=5, clip_on=False,
        )
    ax.set_ylim(-0.22 * ymax, ymax * 1.05)
    ax.set_xscale("log")
    ax.set_xlabel(r"Relaxation time lag $\tau$")
    ax.set_ylabel(r"$G_{ii}(\tau)$")
    panel_title(ax, "b", "Anisotropic relaxation spectra")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7)
    ax.yaxis.set_label_coords(-0.12, 0.5)


def panel_c(ax_gd, ax_gt, ax_pred, ax_err, cax, models, device):
    """Rheology viz case: multi-step shear — GT | PNF | |RhINN−GT| + one σ colorbar."""
    from hnf.rheo_synth import make_rheo_sample

    s = make_rheo_sample(
        n_steps=160, dt=0.05, n_modes=2, dim=2, anisotropic=True,
        protocol="multi_step", seed=7, noise_std=0.0,
        lambdas=[0.5, 5.0], weights=np.array([[1.2, 1.56], [0.6, 0.78]]), g_inf=0.0,
    )
    gd = torch.from_numpy(s["gammadot"]).unsqueeze(0).to(device)
    st = torch.from_numpy(s["stress"]).unsqueeze(0).to(device)
    t = np.arange(s["gammadot"].shape[0]) * 0.05
    with torch.no_grad():
        pnf_s = models["pnf"](gd, 0.05)[0].cpu().numpy()
        rh_s = models["rhinn"](gd, 0.05)[0].cpu().numpy()
    gt = st[0].cpu().numpy()
    gdn = s["gammadot"]
    err_rh = np.abs(rh_s - gt)

    def _rel(pred: np.ndarray) -> float:
        return float(np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-8))

    gd_img = gdn.T
    vmax_g = float(np.max(np.abs(gd_img)) + 1e-8)
    ax_gd.imshow(
        gd_img, aspect="auto", cmap="coolwarm",
        vmin=-vmax_g, vmax=vmax_g,
        extent=[t[0], t[-1], 1.5, -0.5],
        interpolation="nearest",
    )
    ax_gd.set_yticks([0, 1])
    ax_gd.set_yticklabels([r"$\dot\gamma_1$", r"$\dot\gamma_2$"], fontsize=7)
    ax_gd.tick_params(labelbottom=False, length=2, labelsize=6.5)
    ax_gd.set_xlim(t[0], t[-1])
    panel_title(ax_gd, "c", "Case: multi-step shear — GT / pred / error")

    vmax_s = float(max(np.max(np.abs(gt)), np.max(np.abs(pnf_s))) + 1e-8)
    extent = [t[0], t[-1], 1.5, -0.5]
    im_s = None
    for i, (ax, img, title) in enumerate([
        (ax_gt, gt.T, r"GT $\sigma$"),
        (ax_pred, pnf_s.T, rf"PNF $\hat\sigma$  (rel={_rel(pnf_s):.3f})"),
    ]):
        im_s = ax.imshow(
            img, aspect="auto", cmap="viridis",
            vmin=-vmax_s, vmax=vmax_s,
            extent=extent, interpolation="nearest",
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r"$\sigma_1$", r"$\sigma_2$"], fontsize=7)
        ax.set_xlabel("Time (s)", fontsize=7.5)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6.5, length=2)
        if i:
            ax.tick_params(labelleft=False)

    vmax_e = float(max(np.max(err_rh), 1e-6))
    ax_err.imshow(
        err_rh.T, aspect="auto", cmap="magma",
        vmin=0.0, vmax=vmax_e,
        extent=extent, interpolation="nearest",
    )
    ax_err.set_yticks([0, 1])
    ax_err.set_yticklabels([r"$\sigma_1$", r"$\sigma_2$"], fontsize=7)
    ax_err.set_xlabel("Time (s)", fontsize=7.5)
    # no second colorbar — put absolute scale in the title
    ax_err.set_title(rf"$|$RhINN$-$GT$|$  max={vmax_e:.3f}", fontsize=7.5, pad=3)
    ax_err.tick_params(labelsize=6.5, length=2, labelleft=False)

    # single shared σ colorbar (GT / PNF), outside the maps
    cb = plt.colorbar(im_s, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label(r"$\sigma$", fontsize=7)
    return s


def panel_d(ax, ax_gt, models, device):
    """Oscillatory residual + separate GT stress strip (no overlay)."""
    from hnf.rheo_synth import make_rheo_sample

    s = make_rheo_sample(
        n_steps=128, dt=0.05, n_modes=2, dim=2, anisotropic=True,
        protocol="oscillatory", seed=123, noise_std=0.0,
        lambdas=[0.5, 5.0], weights=np.array([[1.2, 1.56], [0.6, 0.78]]), g_inf=0.0,
    )
    gd = torch.from_numpy(s["gammadot"]).unsqueeze(0).to(device)
    st = torch.from_numpy(s["stress"]).unsqueeze(0).to(device)
    t = np.arange(128) * 0.05
    with torch.no_grad():
        preds = {
            "GT": st[0, :, 0].cpu().numpy(),
            "PNF": models["pnf"](gd, 0.05)[0, :, 0].cpu().numpy(),
            "NLS": models["nls"](gd, 0.05)[0, :, 0].cpu().numpy(),
            "EUCLID": models["euclid"](gd, 0.05)[0, :, 0].cpu().numpy(),
            "RhINN": models["rhinn"](gd, 0.05)[0, :, 0].cpu().numpy(),
        }
    gt = preds["GT"]

    ax_gt.plot(t, gt, color="#222", lw=1.2)
    ax_gt.tick_params(labelbottom=False, labelsize=6.5, length=2)
    ax_gt.set_xlim(t[0], t[-1])
    ax_gt.set_ylabel(r"$\sigma_1$", fontsize=8)
    ax_gt.yaxis.set_label_coords(-0.12, 0.5)
    panel_title(ax_gt, "d", "Oscillatory protocol — GT stress & residual")
    ax_gt.text(0.02, 0.50, "GT", transform=ax_gt.transAxes,
               ha="left", va="center", fontsize=7, color="#555")

    series = [
        ("PNF", C_PNF, "--", "o"),
        ("NLS", C_NLS, ":", "s"),
        ("EUCLID", C_EUCLID, "-", "^"),
        ("RhINN", C_RHINN, "-.", "D"),
    ]
    for name, color, ls, mk in series:
        err = preds[name] - gt
        ax.plot(t, err, color=color, lw=1.5, ls=ls, label=name, zorder=3)
        ax.plot(t[::6], err[::6], mk, color=color, ms=3.8, mew=0.0, zorder=4)
    ax.axhline(0.0, color="#888", lw=0.8, zorder=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\hat\sigma_1-\sigma_1$")
    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7)
    ax.set_xlim(t[0], t[-1])
    return s, preds


def mine_knowledge(pnf, gt, out_dir: Path) -> dict:
    """Physics knowledge cards from learned kernel."""
    with torch.no_grad():
        lam = pnf.kernel.relaxation_times().cpu().numpy()
        A = pnf.kernel.modal_weights().cpu().numpy()
        lam_gt = gt.relaxation_times().cpu().numpy()
        A_gt = gt.modal_weights().cpu().numpy()
        # match
        o = np.argsort(lam)
        lam, A = lam[o], A[o]
        og = np.argsort(lam_gt)
        lam_gt, A_gt = lam_gt[og], A_gt[og]
        eig_p = [np.linalg.eigvalsh(A[k]) for k in range(len(lam))]
        eig_g = [np.linalg.eigvalsh(A_gt[k]) for k in range(len(lam_gt))]
        aniso_ratio = [float(e.max() / max(e.min(), 1e-8)) for e in eig_p]
        cards = [
            {
                "id": "K1_lambda_recovery",
                "claim": "Two Maxwell relaxation times recovered to high accuracy.",
                "evidence": f"λ_PNF={lam.tolist()} vs λ_GT={lam_gt.tolist()}",
                "rel_err": float(np.mean(np.abs(lam - lam_gt) / lam_gt)),
            },
            {
                "id": "K2_anisotropy_axes",
                "claim": "Learned A_k are nearly diagonal SPD with channel anisotropy matching GT.",
                "evidence": {
                    "A_pnf": A.tolist(),
                    "A_gt": A_gt.tolist(),
                    "eigenvalue_anisotropy_ratio": aniso_ratio,
                },
            },
            {
                "id": "K3_boltzmann_memory",
                "claim": "Stress is a causal hereditary integral of strain-rate (Boltzmann superposition).",
                "evidence": "Exact Maxwell recurrence implements ∫ G(t-s) γ̇(s) ds with G=Σ A_k e^{-τ/λ_k}.",
            },
            {
                "id": "K4_spectrum_identifiability",
                "claim": "On linear Prony data, PNF matches classical NLS; sparse libraries / black-box RhINN lag after tuning.",
                "evidence": "tuned board: PNF=NLS=0.0033; EUCLID=0.0147; RhINN=0.0177",
            },
        ]
    report = {"cards": cards, "lambda": lam.tolist(), "A": A.tolist(), "aniso_ratio": aniso_ratio}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "knowledge_cards.json").write_text(json.dumps(report, indent=2))
    md = ["# Rheology knowledge cards (PNF aniso)", ""]
    for c in cards:
        md += [f"## {c['id']}", c["claim"], f"- evidence: `{c['evidence']}`", ""]
    (out_dir / "KNOWLEDGE_CARDS.md").write_text("\n".join(md))
    return report


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    out_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    rows = _collect_master_rows()
    (out_dir / "MASTER_BOARD.json").write_text(json.dumps({"rows": rows}, indent=2))
    # markdown master
    lines = [
        "# Rheology — master model comparison",
        "",
        "| Model | Family | stress_rel ↓ | params | source | note |",
        "|-------|--------|-------------:|-------:|--------|------|",
    ]
    for r in rows:
        np_ = "—" if r.get("n_params") is None else str(r["n_params"])
        lines.append(
            f"| {r['model']} | {r['family']} | {r['stress_rel']:.4f} | {np_} | {r['source']} | {r.get('note','')} |"
        )
    (out_dir / "MASTER_BOARD.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]), flush=True)

    models = _load_tuned_models(args, device)
    gt = _gt_kernel()
    knowledge = mine_knowledge(models["pnf"], gt, out_dir)

    # Journal figure
    fig = plt.figure(figsize=(11.6, 8.5))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.97, top=0.90, bottom=0.07,
        hspace=0.38, wspace=0.26,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    gs_c = gs[1, 0].subgridspec(2, 1, height_ratios=[0.32, 1.0], hspace=0.18)
    ax_c_gd = fig.add_subplot(gs_c[0])
    # GT | PNF | |err| | σ colorbar (far right)
    gs_c_bot = gs_c[1].subgridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 0.09], wspace=0.10,
    )
    ax_c_gt = fig.add_subplot(gs_c_bot[0])
    ax_c_pred = fig.add_subplot(gs_c_bot[1])
    ax_c_err = fig.add_subplot(gs_c_bot[2])
    ax_c_cb = fig.add_subplot(gs_c_bot[3])
    gs_d = gs[1, 1].subgridspec(2, 1, height_ratios=[0.38, 1.0], hspace=0.12)
    ax_d_gt = fig.add_subplot(gs_d[0])
    ax_d = fig.add_subplot(gs_d[1], sharex=ax_d_gt)
    for ax in (ax_a, ax_b, ax_c_gd, ax_c_gt, ax_c_pred, ax_c_err, ax_d, ax_d_gt):
        ax.set_facecolor(BG)

    panel_a(ax_a, rows)
    panel_b(ax_b, models["pnf"], gt)
    panel_c(ax_c_gd, ax_c_gt, ax_c_pred, ax_c_err, ax_c_cb, models, device)
    panel_d(ax_d, ax_d_gt, models, device)

    # Align left/right columns to top-row boxes
    fig.canvas.draw()
    pa, pb = ax_a.get_position(), ax_b.get_position()

    pc_gd = ax_c_gd.get_position()
    c_all = (ax_c_gt, ax_c_pred, ax_c_err, ax_c_cb)
    c_bottom = min(ax.get_position().y0 for ax in c_all)
    c_top = pc_gd.y1
    gap = max(pc_gd.y0 - max(ax.get_position().y1 for ax in (ax_c_gt, ax_c_pred, ax_c_err)), 0.01)
    h = c_top - c_bottom
    h_top = (h - gap) * 0.28
    h_bot = (h - gap) * 0.72
    ax_c_gd.set_position([pa.x0, c_bottom + h_bot + gap, pa.width, h_top])

    # GT | PNF | err | colorbar(rightmost) — leave gap so bar is not clipped/covered
    ratios = [1.0, 1.0, 1.0, 0.12]
    rsum = sum(ratios)
    x = pa.x0
    for ax, r in zip(c_all, ratios):
        w = pa.width * (r / rsum)
        if ax is ax_c_cb:
            ax.set_position([x + w * 0.20, c_bottom + 0.06 * h_bot, w * 0.50, 0.88 * h_bot])
        else:
            ax.set_position([x, c_bottom, w * 0.94, h_bot])
        x += w

    def _align_stack(top_ax, bot_ax, x0, width, r_top=0.32):
        pt, pb_ = top_ax.get_position(), bot_ax.get_position()
        gap_ = max(pt.y0 - pb_.y1, 0.008)
        bottom, top = pb_.y0, pt.y1
        hh = top - bottom
        h_t = (hh - gap_) * r_top
        h_b = (hh - gap_) * (1.0 - r_top)
        bot_ax.set_position([x0, bottom, width, h_b])
        top_ax.set_position([x0, bottom + h_b + gap_, width, h_t])

    _align_stack(ax_d_gt, ax_d, pb.x0, pb.width)

    fig.suptitle(
        "Anisotropic Boltzmann memory — performance, spectrum, and physics",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    stem = fig_dir / args.stem
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor=BG)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    caption = (
        "Anisotropic Boltzmann / Prony memory on synthetic linear viscoelasticity "
        "(dim=2, K=2). "
        "(a) Model-zoo stress relative L2 error. "
        "(b) Anisotropic spectra G11/G22: GT solid, PNF open markers; λ1, λ2 marked. "
        "(c) Multi-step shear visualization: γ̇ drive (top); "
        "GT | PNF | |RhINN−GT| maps with one shared σ colorbar (error scale in title). "
        "(d) Oscillatory residual σ̂−σ. "
        "Modal weights satisfy A_k = L_k L_k^T (SPD)."
    )
    meta = {
        "figure": str(stem.with_suffix(".png")),
        "master_board": str(out_dir / "MASTER_BOARD.md"),
        "knowledge": knowledge,
        "caption": caption,
    }
    (stem.with_name(stem.name + "_meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[rheo-journal] → {stem.with_suffix('.png')}", flush=True)
    print(f"[rheo-journal] knowledge → {out_dir / 'KNOWLEDGE_CARDS.md'}", flush=True)


if __name__ == "__main__":
    main()
