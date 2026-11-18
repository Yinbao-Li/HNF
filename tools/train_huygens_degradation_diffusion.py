#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal trainable Huygens physical-degradation diffusion (new paper / HDD).

  python tools/train_huygens_degradation_diffusion.py --device cuda --epochs 2 \
    --samples-per-epoch 8000 --output-dir outputs/hdd_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from hnf.huygens_degradation import HuygensDegradationDiffusion  # noqa: E402
from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.train_stead_picking import move_batch_to_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/hdd_smoke"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--samples-per-epoch", type=int, default=8000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=4)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--mode", choices=["huygens", "gaussian"], default="huygens")
    p.add_argument("--freeze-huygens", action="store_true", default=False)
    p.add_argument("--max-train-events", type=int, default=20000)
    p.add_argument("--max-train-noise", type=int, default=5000)
    p.add_argument("--max-val", type=int, default=1000)
    return p.parse_args()


@torch.no_grad()
def eval_recon(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        x = batch["x"]
        out = model(x)
        mse = torch.mean((out["x_hat"] - x) ** 2).item()
        tot += mse * x.size(0)
        n += x.size(0)
    return {"mse": tot / max(n, 1)}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/hdd/smoke")
    results_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = HuygensDegradationDiffusion(
        timesteps=args.timesteps,
        dim=args.dim,
        mode=args.mode,
        freeze_huygens=args.freeze_huygens,
        n_huygens_blocks=2,
    ).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[hdd] mode={args.mode} params={n_all:,} trainable={n_tr:,} T={args.timesteps} device={device}", flush=True)

    train_ds = STEADPickingDataset(
        "train",
        seq_len=6000,
        max_event_traces=args.max_train_events,
        max_noise_traces=args.max_train_noise,
        seed=args.seed,
        augment=True,
        fresh_augment=True,
        aug_real_noise_prob=0.5,
        aug_real_noise_snr_db=(-5.0, 15.0),
        aug_gauss_prob=0.3,
        aug_noise_snr_db=(5.0, 20.0),
        load_geometry=False,
    )
    val_ds = STEADPickingDataset(
        "val",
        seq_len=6000,
        max_event_traces=args.max_val // 2,
        max_noise_traces=args.max_val // 2,
        seed=args.seed,
        load_geometry=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_ds, replacement=True, num_samples=args.samples_per_epoch),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    history = []
    best = {"mse": 1e9}

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run, n = 0.0, 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            x = batch["x"]
            # demean/std like picker
            x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = model.reconstruction_loss(x, out)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * x.size(0)
            n += x.size(0)
        tr_loss = run / max(n, 1)
        val = eval_recon(model, val_loader, device)
        row = {"epoch": ep, "train_loss": tr_loss, "val_mse": val["mse"], "sec": time.time() - t0}
        history.append(row)
        print(
            f"ep{ep:02d} loss={tr_loss:.4f} val_mse={val['mse']:.4f} ({row['sec']:.0f}s)",
            flush=True,
        )
        ckpt = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "args": vars(args),
            "val": val,
            "history": history,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val["mse"] < best["mse"]:
            best = {"mse": val["mse"], "epoch": ep}
            torch.save(ckpt, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    summary = {
        "project": "huygens_degradation_diffusion",
        "paper": "HDD (new; not TGRS)",
        "mode": args.mode,
        "best": best,
        "history": history,
        "n_params": n_all,
        "n_trainable": n_tr,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "README.md").write_text(
        "# HDD smoke\n\nMinimal Huygens physical-degradation diffusion train.\n"
        f"mode={args.mode} best_val_mse={best['mse']:.4f} ep={best.get('epoch')}\n"
    )
    print(f"[done] best={best} → {args.output_dir} / {results_dir}", flush=True)


if __name__ == "__main__":
    main()
