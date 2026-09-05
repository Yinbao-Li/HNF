#!/usr/bin/env python
"""RadHAR v4: ablation-informed wave + WaveGRU hybrid.

Changes over v3 (ablation-informed):
  - EnergyPool removed → mean-only head  (-EnergyPool gave +0.11pp)
  - learnable_kernel_params=False         (fixed-kernel gave +0.09pp)
  - MixUp removed                         (-MixUp gave +0.03pp)
  - ConvStem kept                         (-ConvStem lost -0.20pp)
  - ClassWeight kept                      (-ClassWeight lost -0.20pp)

New in v4:
  - --model wave_gru: WaveGRU hybrid (wave feature extractor → BiGRU)
  - --fft: append FFT magnitude spectrum as extra channels
  - --t-steps 120: longer window option (better for jack/jump periodicity)
  - 60 epochs (same as v3)

Usage:
  # wave only (ablation-optimised)
  python tools/train_radhar_huygens_cls_v4.py --device cuda --model wave

  # WaveGRU hybrid
  python tools/train_radhar_huygens_cls_v4.py --device cuda --model wave_gru

  # WaveGRU + longer window
  python tools/train_radhar_huygens_cls_v4.py --device cuda --model wave_gru --t-steps 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.radhar_cls_dataset import ACTIVITY_TO_IDX, RadHARClsDataset, class_weights_from_samples
from hnf.radhar_huygens_cls import RadHARHuygensClsModel, WaveGRUModel
from hnf.radhar_io import ACTIVITIES

N_CLASSES = len(ACTIVITIES)


# ─────────────────────────── FFT augment ─────────────────────────────────────

def append_fft_channels(x: torch.Tensor) -> torch.Tensor:
    """Append FFT magnitude of each channel as extra temporal feature.

    x: (B, C, T) → (B, C*2, T)  [original + freq-domain magnitude resampled to T]
    Only uses first T//2 FFT bins resampled to T via linear interp.
    """
    B, C, T = x.shape
    # FFT over time axis
    X_f = torch.fft.rfft(x, dim=-1)          # (B, C, T//2+1)
    mag = X_f.abs()                           # (B, C, T//2+1)
    # resample to T via linear interpolation
    mag_r = F.interpolate(mag, size=T, mode="linear", align_corners=False)  # (B, C, T)
    return torch.cat([x, mag_r], dim=1)       # (B, 2C, T)


# ─────────────────────────── LR ──────────────────────────────────────────────

def cosine_lr(epoch: int, total: int, base_lr: float, warmup: int = 3) -> float:
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ─────────────────────────── EVAL ────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, class_w, use_fft: bool) -> dict:
    model.eval()
    ys, preds = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        if use_fft:
            x = append_fft_channels(x)
        t = batch["t"].to(device)
        y = batch["y"].to(device)
        logits = model(x, t)["logits"]
        total_loss += float(F.cross_entropy(logits, y, weight=class_w, reduction="sum").cpu())
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
        n += y.numel()
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    acc = float((y_all == p_all).mean())
    per = {
        name: float((p_all[y_all == idx] == idx).mean()) if (y_all == idx).any() else float("nan")
        for name, idx in ACTIVITY_TO_IDX.items()
    }
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for yi, pi in zip(y_all, p_all):
        cm[int(yi), int(pi)] += 1
    return {"loss": total_loss / max(n, 1), "accuracy": acc,
            "per_class_recall": per, "n": n, "confusion": cm.tolist()}


# ─────────────────────────── ARGS ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/radhar/Data"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/mmwave/radhar_cls_cache"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-range-bins", type=int, default=24)
    p.add_argument("--n-doppler-bins", type=int, default=8)
    p.add_argument("--t-steps", type=int, default=60)
    p.add_argument("--stride-frames", type=int, default=10)
    p.add_argument("--stem-ch", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--model", choices=["wave", "wave_gru"], default="wave_gru")
    p.add_argument("--fft", action="store_true", help="Append FFT magnitude as extra channels")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prev-summary", type=Path,
                   default=Path("outputs/mmwave/radhar_huygens_cls_v3/SUMMARY.json"))
    return p.parse_args()


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    tag = f"{args.model}_t{args.t_steps}{'_fft' if args.fft else ''}"
    if args.output_dir is None:
        args.output_dir = Path(f"outputs/mmwave/radhar_huygens_cls_v4_{tag}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds_kw = dict(
        n_range_bins=args.n_range_bins, t_steps=args.t_steps,
        stride_frames=args.stride_frames, feature_mode="rich",
        include_range_doppler=True, n_doppler_bins=args.n_doppler_bins,
        cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache,
    )
    train_ds = RadHARClsDataset(args.data_dir, "train", augment=True, seed=args.seed, **ds_kw)
    test_ds  = RadHARClsDataset(args.data_dir, "test",  augment=False, **ds_kw)

    n_ch_raw = train_ds.n_channels
    n_ch = n_ch_raw * 2 if args.fft else n_ch_raw  # FFT doubles channels

    class_w = class_weights_from_samples(train_ds.samples).to(device)
    for act, boost in (("jump", 1.5), ("jack", 1.25)):
        class_w[ACTIVITY_TO_IDX[act]] *= boost
    class_w = class_w / class_w.mean()

    epoch_sec = train_ds.epoch_sec
    local_win = min(0.5, epoch_sec * 0.4)

    wave_kw = dict(
        n_channels=n_ch, n_classes=N_CLASSES,
        stem_ch=args.stem_ch, embed_dim=args.embed_dim,
        num_shared_layers=args.num_layers,
        epoch_sec=epoch_sec, local_window_sec=local_win,
        dropout=0.3, medium_hidden=64,
        learnable_kernel_params=False,   # ablation: fixed kernel wins
        use_energy_pool=False,           # ablation: mean-only wins
        residual_energy=False,
    )

    if args.model == "wave":
        model = RadHARHuygensClsModel(**wave_kw).to(device)
    else:  # wave_gru
        model = WaveGRUModel(
            n_channels=n_ch, n_classes=N_CLASSES,
            stem_ch=args.stem_ch, embed_dim=args.embed_dim,
            num_wave_layers=args.num_layers,
            gru_hidden=args.gru_hidden, gru_layers=args.gru_layers,
            epoch_sec=epoch_sec, local_window_sec=local_win,
            dropout=0.3, medium_hidden=64,
            learnable_kernel_params=False,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v4/{tag}] train={len(train_ds)} test={len(test_ds)} "
          f"C={n_ch} T={args.t_steps} model={args.model} fft={args.fft} params={n_params:,}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best = [], {"accuracy": -1.0}

    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(epoch, args.epochs, args.lr, warmup=3)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        run_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            if args.fft:
                x = append_fft_channels(x)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            logits = model(x, t)["logits"]
            loss = F.cross_entropy(logits, y, weight=class_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach().cpu()) * y.size(0)
            n += y.size(0)

        te = evaluate(model, test_loader, device, class_w, args.fft)
        row = {"epoch": epoch, "lr": lr, "train_loss": run_loss / max(n, 1),
               "test_accuracy": te["accuracy"], "per_class_recall": te["per_class_recall"]}
        history.append(row)
        print(f"epoch {epoch:02d} lr={lr:.2e} tr={row['train_loss']:.4f} "
              f"acc={te['accuracy']:.4f} "
              f"jack={te['per_class_recall'].get('jack', 0):.3f} "
              f"jump={te['per_class_recall'].get('jump', 0):.3f}",
              flush=True)

        if te["accuracy"] > best["accuracy"]:
            best = {"accuracy": te["accuracy"], "epoch": epoch,
                    "per_class_recall": te["per_class_recall"],
                    "confusion": te["confusion"], "n_test": te["n"]}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "n_channels": n_ch, "n_channels_raw": n_ch_raw},
                       args.output_dir / "best.pt")

    prev_acc = None
    if args.prev_summary.exists():
        try:
            prev_acc = json.loads(args.prev_summary.read_text()).get("best", {}).get("accuracy")
        except Exception:
            pass

    summary = {
        "dataset": "RadHAR", "version": f"v4_{tag}",
        "method": f"{args.model} + fft={args.fft}",
        "n_train": len(train_ds), "n_test": len(test_ds),
        "n_channels": n_ch, "n_params": n_params,
        "t_steps": args.t_steps,
        "best": best,
        "v3_baseline_accuracy": prev_acc,
        "delta_vs_v3": (best["accuracy"] - prev_acc) if prev_acc is not None else None,
        "history": history,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = [
        f"# RadHAR v4 / {tag}",
        "",
        f"Params: {n_params:,} | Best acc: **{best['accuracy']:.4f}** (epoch {best.get('epoch')})",
    ]
    if prev_acc is not None:
        lines.append(f"v3 baseline: {prev_acc:.4f} | delta: **{best['accuracy'] - prev_acc:+.4f}**")
    lines += ["", "## Per-class recall (best epoch)", ""]
    for k, v in (best.get("per_class_recall") or {}).items():
        lines.append(f"- {k}: {v:.4f}")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\n[done] best_acc={best['accuracy']:.4f}  delta_vs_v3="
          f"{best['accuracy'] - prev_acc if prev_acc else 0:+.4f}  → {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
