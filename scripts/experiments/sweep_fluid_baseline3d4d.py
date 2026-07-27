#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/eval 3D/4D baselines (U-Net, CNN-AE) and literature SOTA (RecFNO, FlowMRI-Net)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.fluid_baselines3d import Conv3DAutoencoder, UNet3DReconstructor, UNet4DReconstructor
from hnf.fluid_sota3d import FlowMRINetUnrolled3D, RecFNO3D
from hnf.fluid_dataset3d import SyntheticFluid3DDataset
from hnf.fluid_dataset4d import SyntheticFluid4DDataset
from hnf.fluid_losses import rel_err_masked, velocity_recon_loss
from tools.train_fluid import set_seed


def train_eval_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"ep{epoch}", leave=False):
            x = batch["x"].to(device)
            dense = batch["dense"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _ = velocity_recon_loss(pred, dense, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    rels, by_fam = [], {}
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            dense = batch["dense"].to(device)
            pred = model(x)
            for i in range(x.size(0)):
                r = rel_err_masked(pred[i], dense[i])
                rels.append(r)
                fam = str(batch["family"][i])
                by_fam.setdefault(fam, []).append(r)
    out = {"vel_rel": float(sum(rels) / max(len(rels), 1)), "n_params": sum(p.numel() for p in model.parameters())}
    for k, v in by_fam.items():
        out[f"vel_rel_{k}"] = float(sum(v) / len(v))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--output", default="outputs/fluid/baseline3d4d_board.json")
    p.add_argument("--skip-baselines", action="store_true", help="only train literature SOTA models")
    p.add_argument("--skip-literature", action="store_true", help="only train U-Net/CNN baselines")
    args = p.parse_args()
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3D vortex (headline task) ---
    kw3 = dict(d=12, h=12, w=12, keep_frac=0.1, seed=42, families=["vortex_tube"])
    train3 = DataLoader(SyntheticFluid3DDataset("train", 512, **kw3), batch_size=4, shuffle=True)
    test3 = DataLoader(SyntheticFluid3DDataset("test", 128, **kw3), batch_size=4)

    board: dict = {"settings": {"keep_frac": 0.1, "grid3d": "12^3", "epochs": args.epochs}}

    if out_path.is_file() and (args.skip_baselines or args.skip_literature):
        board.update(json.loads(out_path.read_text()))

    if not args.skip_baselines:
        for name, model in [
            ("unet3d", UNet3DReconstructor(base=24)),
            ("cnn3d_ae", Conv3DAutoencoder(width=32)),
        ]:
            print(f"[baseline] training {name} 3D vortex...", flush=True)
            board[name] = train_eval_baseline(model, train3, test3, device, args.epochs, lr=3e-4)

    # HNF from completed run
    hnf_path = _REPO / "outputs/fluid/spatial_3d4d_suite/synth3d_vortex/summary.json"
    if hnf_path.is_file():
        s = json.loads(hnf_path.read_text())
        board["hnf_spatial3d_rot"] = {
            "vel_rel": s["test_best"]["vel_rel"],
            "vel_rel_vortex_tube": s["test_best"].get("vel_rel_vortex_tube", s["test_best"]["vel_rel"]),
            "n_params": 71027,
            "source": str(hnf_path),
        }

    # prior 2D spatial
    p2 = _REPO / "outputs/fluid/spatial_suite/vortex50/summary.json"
    if p2.is_file():
        s2 = json.loads(p2.read_text())
        board["hnf_spatial2d_rot"] = {"vel_rel": s2["test_best"]["vel_rel"], "n_params": ~65000}

    board["raster_hnf_1d"] = {"vel_rel": 0.87, "note": "Stage-0 raster baseline on vortex (from prior ablation)"}

    if not args.skip_baselines:
        kw_all = dict(d=12, h=12, w=12, keep_frac=0.1, seed=42)
        train_a = DataLoader(SyntheticFluid3DDataset("train", 512, **kw_all), batch_size=4, shuffle=True)
        test_a = DataLoader(SyntheticFluid3DDataset("test", 128, **kw_all), batch_size=4)
        board["unet3d_all"] = train_eval_baseline(UNet3DReconstructor(base=24), train_a, test_a, device, args.epochs, lr=3e-4)

        kw4 = dict(t_steps=4, d=8, h=12, w=12, keep_frac=0.1, seed=42)
        train4 = DataLoader(SyntheticFluid4DDataset("train", 256, **kw4), batch_size=2, shuffle=True)
        test4 = DataLoader(SyntheticFluid4DDataset("test", 64, **kw4), batch_size=2)
        print("[baseline] training unet4d...", flush=True)
        board["unet4d"] = train_eval_baseline(
            UNet4DReconstructor(t_steps=4, base=16), train4, test4, device, max(args.epochs, 20), lr=2e-4
        )

    if not args.skip_literature:
        board.setdefault("literature_refs", {})
        board["literature_refs"]["recfno3d"] = "Zhao et al., 2023, Int. J. Thermal Sciences (arXiv:2302.09808)"
        board["literature_refs"]["flowmri_net3d"] = "Wallerberger et al., 2025, JOCMR (arXiv:2410.08856)"
        for name, model in [
            ("recfno3d", RecFNO3D(width=24, modes=4)),
            ("flowmri_net3d", FlowMRINetUnrolled3D(n_stages=6, base=24)),
        ]:
            print(f"[literature] training {name} 3D vortex...", flush=True)
            board[name] = train_eval_baseline(model, train3, test3, device, args.epochs, lr=3e-4)
        kw_all = dict(d=12, h=12, w=12, keep_frac=0.1, seed=42)
        train_a = DataLoader(SyntheticFluid3DDataset("train", 512, **kw_all), batch_size=4, shuffle=True)
        test_a = DataLoader(SyntheticFluid3DDataset("test", 128, **kw_all), batch_size=4)
        for name, model in [
            ("recfno3d_all", RecFNO3D(width=24, modes=4)),
            ("flowmri_net3d_all", FlowMRINetUnrolled3D(n_stages=6, base=24)),
        ]:
            print(f"[literature] training {name}...", flush=True)
            board[name] = train_eval_baseline(model, train_a, test_a, device, args.epochs, lr=3e-4)

    # --- 3D all families (HNF checkpoint) ---
    hnf_all = _REPO / "outputs/fluid/spatial_3d4d_suite/synth3d_all/summary.json"
    if hnf_all.is_file():
        sa = json.loads(hnf_all.read_text())
        board["hnf_spatial3d_all"] = {"vel_rel": sa["test_best"]["vel_rel"], **{k: v for k, v in sa["test_best"].items() if k.startswith("vel_rel_")}}

    # --- 4D HNF checkpoint ---
    hnf4 = _REPO / "outputs/fluid/spatial_3d4d_suite/synth4d_all/summary.json"
    if hnf4.is_file():
        s4 = json.loads(hnf4.read_text())
        board["hnf_spatial4d"] = {"vel_rel": s4["test_best"]["vel_rel"], **{k: v for k, v in s4["test_best"].items() if k.startswith("vel_rel_")}}

    out_path.write_text(json.dumps(board, indent=2))
    md = _REPO / "outputs/fluid/BASELINE3D4D.md"
    lines = [
        "# Fluid 3D/4D baseline comparison (@10% keep)",
        "",
        "U-Net and CNN-AE are **standard baselines**, not literature SOTA.",
        "",
        "## 3D vortex_tube (12³)",
        "",
        "| Model | test vel_rel | params |",
        "|-------|-------------:|-------:|",
    ]
    for key in ["unet3d", "cnn3d_ae", "recfno3d", "flowmri_net3d", "hnf_spatial3d_rot", "hnf_spatial2d_rot", "raster_hnf_1d"]:
        if key in board and isinstance(board[key], dict) and "vel_rel" in board[key]:
            r = board[key]["vel_rel"]
            n = board[key].get("n_params", "—")
            lines.append(f"| {key} | {r:.4f} | {n} |")
    lines += [
        "",
        "RecFNO = Zhao et al. 2023; FlowMRI-Net = Wallerberger et al. 2025 (grid-domain simplified).",
        "",
        "## 3D all families",
        "",
        "| Model | vel_rel |",
        "|-------|--------:|",
    ]
    for key in ["unet3d_all", "recfno3d_all", "flowmri_net3d_all", "hnf_spatial3d_all"]:
        if key in board:
            lines.append(f"| {key} | {board[key]['vel_rel']:.4f} |")
    if "unet4d" in board:
        lines += ["", "## 4D synth (8×12×12, T=4)", "", "| Model | vel_rel |", "|-------|--------:|"]
        lines.append(f"| unet4d | {board['unet4d']['vel_rel']:.4f} |")
        if "hnf_spatial4d" in board:
            lines.append(f"| hnf_spatial4d | {board['hnf_spatial4d']['vel_rel']:.4f} |")
    md.write_text("\n".join(lines) + "\n")
    print(f"[baseline] → {out_path}  {md}", flush=True)


if __name__ == "__main__":
    main()
