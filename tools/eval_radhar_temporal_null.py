#!/usr/bin/env python
"""B3 — RadHAR temporal nulls for NMI Paper B (frame shuffle / history truncate).

Eval-only on a trained Wave+Pattern checkpoint. Within-frame points stay
co-temporal; we only break **inter-frame** order or shorten history.

  python tools/eval_radhar_temporal_null.py --device cuda \\
      --checkpoint outputs/mmwave/radhar_wave_pattern_t120/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hnf.radhar_cls_dataset import RadHARClsDataset
from hnf.radhar_io import ACTIVITIES
from hnf.radhar_wave_pattern_cls import WavePlusPatternClsModel
from hnf.radhar_huygens_cls import build_time_axis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/radhar/Data"))
    p.add_argument("--cache-dir", type=Path, default=Path("outputs/mmwave/radhar_cls_cache"))
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/mmwave/radhar_wave_pattern_t120/best.pt"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mmwave/radhar_temporal_null_b3"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--t-steps", type=int, default=120)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _acc_from_logits(logits, y):
    pred = logits.argmax(-1)
    return float((pred == y).float().mean().cpu())


@torch.no_grad()
def eval_mode(model, loader, device, *, mode: str, t_keep: int | None, epoch_sec: float):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        x = batch["x"].to(device)  # (B, C, T) or (B, T, C)?
        # dataset uses (B, T, C) in train script: batch["x"] then model(x,t)
        t = batch["t"].to(device)
        y = batch["y"].to(device)

        # x: (B, C, T) — only corrupt the frame/time axis T
        if mode == "clean":
            x_use, t_use = x, t
        elif mode == "frame_shuffle":
            b, _, t_len = x.shape
            x_use = torch.empty_like(x)
            for i in range(b):
                perm = torch.randperm(t_len, device=device)
                x_use[i] = x[i, :, perm]
            t_use = build_time_axis(t_len, epoch_sec, device=device).view(1, t_len, 1).expand(b, -1, -1)
        elif mode == "time_reverse":
            x_use = torch.flip(x, dims=[-1])
            t_use = t  # ascending t on reversed frames → anti-causal mismatch
        elif mode == "history_trunc":
            assert t_keep is not None and t_keep < x.size(-1)
            x_use = torch.zeros_like(x)
            x_use[:, :, -t_keep:] = x[:, :, -t_keep:]
            t_use = t
        else:
            raise ValueError(mode)

        logits = model(x_use, t_use)["logits"]
        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    acc = float((y_all == p_all).mean())
    per = {}
    for i, name in enumerate(ACTIVITIES):
        m = y_all == i
        per[name] = float((p_all[m] == i).mean()) if m.any() else float("nan")
    return {"accuracy": acc, "per_class_recall": per, "n": int(len(y_all)), "mode": mode, "t_keep": t_keep}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_ds = RadHARClsDataset(
        args.data_dir, split="test", cache_dir=args.cache_dir, t_steps=args.t_steps, seed=args.seed
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    # Match train_radhar_wave_pattern_cls.py defaults (T=120 → epoch_sec=4).
    model = WavePlusPatternClsModel().to(device)
    model.load_state_dict(state, strict=True)
    epoch_sec = float(getattr(model, "epoch_sec", 4.0))

    rows = []
    for mode, t_keep in [
        ("clean", None),
        ("frame_shuffle", None),
        ("time_reverse", None),
        ("history_trunc", 60),
        ("history_trunc", 30),
    ]:
        tag = mode if t_keep is None else f"history_trunc_T{t_keep}"
        print(f"\n=== {tag} ===", flush=True)
        r = eval_mode(model, loader, device, mode=mode, t_keep=t_keep, epoch_sec=epoch_sec)
        r["id"] = tag
        rows.append(r)
        print(f"  acc={r['accuracy']:.4f} n={r['n']}", flush=True)

    clean = next(r for r in rows if r["id"] == "clean")["accuracy"]
    shuf = next(r for r in rows if r["id"] == "frame_shuffle")["accuracy"]
    rev = next(r for r in rows if r["id"] == "time_reverse")["accuracy"]
    t60 = next(r for r in rows if r["id"] == "history_trunc_T60")["accuracy"]
    t30 = next(r for r in rows if r["id"] == "history_trunc_T30")["accuracy"]

    # Pass criteria (Paper B protocol Null A)
    drop_shuf = clean - shuf
    gate_shuffle = drop_shuf >= 0.05  # ≥5 percentage points
    gate_history = (clean - t30) >= 0.02 and t30 <= t60 <= clean + 1e-6
    summary = {
        "checkpoint": str(args.checkpoint),
        "clean_acc": clean,
        "frame_shuffle_acc": shuf,
        "time_reverse_acc": rev,
        "history_T60_acc": t60,
        "history_T30_acc": t30,
        "delta_shuffle": drop_shuf,
        "pass_frame_shuffle_gate": gate_shuffle,
        "pass_history_gate": gate_history,
        "pass_criteria": {
            "frame_shuffle": "clean - shuffle >= 0.05",
            "history": "clean >= T60 >= T30 and clean - T30 >= 0.02",
        },
        "rows": rows,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# RadHAR temporal null (NMI Paper B / B3)",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        "",
        f"| Condition | Acc | Δ vs clean |",
        f"|---|---:|---:|",
        f"| clean | **{clean:.4f}** | — |",
        f"| frame_shuffle | {shuf:.4f} | {drop_shuf:+.4f} |",
        f"| time_reverse | {rev:.4f} | {rev-clean:+.4f} |",
        f"| history T=60 | {t60:.4f} | {t60-clean:+.4f} |",
        f"| history T=30 | {t30:.4f} | {t30-clean:+.4f} |",
        "",
        f"- frame_shuffle gate (≥5pp drop): **{'PASS' if gate_shuffle else 'FAIL'}**",
        f"- history gate: **{'PASS' if gate_history else 'FAIL'}**",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
