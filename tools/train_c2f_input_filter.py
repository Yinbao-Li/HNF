#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a waveform front-end filter; freeze c2f picker; pick on filtered traces.

  x' = x + softplus(α)·CNN(x)   (CNN zero-init → identity at start)
  ŷ = M_frozen(x')

Pilot:
  python tools/train_c2f_input_filter.py --device cuda --epochs 3 \
    --samples-per-epoch 40000 --output-dir outputs/stead_c2f_input_filter_pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from hnf.stead_picking_dataset import STEADPickingDataset  # noqa: E402
from tools.sweep_stead_c2f_threshold import load_model  # noqa: E402
from tools.train_stead_c2f_picking import (  # noqa: E402
    asym_p_target,
    evaluate,
    gaussian_target,
    schedule_lr,
)
from tools.train_stead_picking import move_batch_to_device  # noqa: E402


class WaveformResidualFilter(nn.Module):
    """Near-identity residual Conv filter on (B,T,C)."""

    def __init__(self, channels: int = 3, hidden: int = 24, kernel: int = 15, layers: int = 4):
        super().__init__()
        pad = kernel // 2
        blocks = []
        in_ch = channels
        for i in range(layers - 1):
            blocks += [
                nn.Conv1d(in_ch, hidden, kernel, padding=pad, padding_mode="reflect"),
                nn.GELU(),
            ]
            in_ch = hidden
        self.body = nn.Sequential(*blocks)
        self.out = nn.Conv1d(hidden, channels, kernel, padding=pad, padding_mode="reflect")
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.alpha = nn.Parameter(torch.tensor(-2.0))  # softplus(-2)≈0.13

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C) → conv (B,C,T)
        h = x.transpose(1, 2)
        delta = self.out(self.body(h))
        delta = delta - delta.mean(-1, keepdim=True)  # no DC inject
        y = h + F.softplus(self.alpha) * delta
        return y.transpose(1, 2).contiguous()


class FilteredC2F(nn.Module):
    def __init__(self, picker: nn.Module, filt: WaveformResidualFilter):
        super().__init__()
        self.picker = picker
        self.filter = filt

    def forward(self, x, t=None, **kw):
        x_f = self.filter(x)
        return self.picker(x_f, t, **kw)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/stead_c2f_imp_RSP/last.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stead_c2f_input_filter_pilot"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--samples-per-epoch", type=int, default=40000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-val", type=int, default=4000)
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--pick-tolerance-sec", type=float, default=0.5)
    p.add_argument("--label-sigma-sec", type=float, default=0.35)
    p.add_argument("--late-sigma-ratio", type=float, default=0.4)
    p.add_argument("--manual-weight", type=float, default=2.0)
    p.add_argument("--pick-pos-weight", type=float, default=1.0)
    p.add_argument("--pick-loss-weight", type=float, default=2.0)
    p.add_argument("--det-loss-weight", type=float, default=1.0)
    p.add_argument("--noise-loss-weight", type=float, default=0.15)
    p.add_argument("--noise-max-weight", type=float, default=0.1)
    p.add_argument("--wave-sup-weight", type=float, default=0.5)
    p.add_argument("--wave-pos-weight", type=float, default=3.0)
    p.add_argument("--refine-sup-weight", type=float, default=0.0)  # picker frozen; skip refine aux
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--filter-hidden", type=int, default=24)
    p.add_argument("--filter-layers", type=int, default=4)
    p.add_argument("--aug-real-noise-prob", type=float, default=0.8)
    p.add_argument("--aug-real-noise-snr-min", type=float, default=-5.0)
    p.add_argument("--aug-real-noise-snr-max", type=float, default=10.0)
    p.add_argument("--skip-snr-eval", action="store_true")
    return p.parse_args()


def train_epoch(model, loader, opt, device, args, seq_len, epoch):
    model.train()
    model.picker.eval()  # frozen picker: keep BN/dropout eval
    lr = schedule_lr(epoch, args.epochs, args.lr, warmup=args.warmup, kind="cosine")
    for pg in opt.param_groups:
        pg["lr"] = lr
    pos_w = torch.tensor(args.pick_pos_weight, device=device)
    sigma_bins = max(1.0, args.label_sigma_sec * seq_len / 60.0)
    use_amp = bool(args.amp and device.type == "cuda")
    run, n = 0.0, 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = model(batch["x"], batch["t"])
            det_loss = F.binary_cross_entropy_with_logits(out["det"].float(), batch["det"].float())
            if args.late_sigma_ratio < 1.0:
                p_tgt = asym_p_target(batch, seq_len, sigma_bins, args.late_sigma_ratio, device)
            else:
                p_tgt = batch.get("p_target", gaussian_target(batch["p_idx"], seq_len, sigma_bins, device)).float()
            s_tgt = batch.get("s_target", gaussian_target(batch["s_idx"], seq_len, sigma_bins, device)).float()
            p_mask = (batch.get("p_mask", batch["p_valid"]) > 0).float().unsqueeze(-1)
            if args.manual_weight != 1.0 and "p_manual" in batch:
                p_mask = p_mask * (1.0 + (args.manual_weight - 1.0) * batch["p_manual"].float()).unsqueeze(-1)
            s_mask = (batch.get("s_mask", batch["s_valid"]) > 0).float().unsqueeze(-1)
            p_loss = F.binary_cross_entropy_with_logits(out["p"], p_tgt, reduction="none")
            s_loss = F.binary_cross_entropy_with_logits(out["s"], s_tgt, reduction="none")
            p_loss = (p_loss * (1.0 + (pos_w - 1.0) * p_tgt) * p_mask).mean()
            s_loss = (s_loss * (1.0 + (pos_w - 1.0) * s_tgt) * s_mask).mean()
            noise = (batch["det"] < 0.5).float().unsqueeze(-1)
            loss = (
                args.det_loss_weight * det_loss
                + args.pick_loss_weight * (p_loss + s_loss)
                + args.noise_loss_weight
                * ((F.softplus(out["p"]) * noise).mean() + (F.softplus(out["s"]) * noise).mean())
            )
            if args.wave_sup_weight > 0:
                wpos = args.wave_pos_weight if args.wave_pos_weight > 0 else args.pick_pos_weight
                pw = F.binary_cross_entropy_with_logits(out["p_wave"].float(), p_tgt, reduction="none")
                sw = F.binary_cross_entropy_with_logits(out["s_wave"].float(), s_tgt, reduction="none")
                pw = (pw * (1.0 + (wpos - 1.0) * p_tgt) * p_mask).mean()
                sw = (sw * (1.0 + (wpos - 1.0) * s_tgt) * s_mask).mean()
                loss = loss + args.wave_sup_weight * (pw + sw)
            if args.noise_max_weight > 0:
                nz = noise.squeeze(-1)
                pm = F.softplus(out["p"].float().amax(-1))
                sm = F.softplus(out["s"].float().amax(-1))
                loss = loss + args.noise_max_weight * ((pm + sm) * nz).sum() / nz.sum().clamp_min(1.0)
            # identity regularizer: keep filter small on clean-ish batches
            x_f = model.filter(batch["x"])
            id_pen = ((x_f - batch["x"]) ** 2).mean()
            loss = loss + 0.05 * id_pen
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.filter.parameters(), args.grad_clip)
        opt.step()
        bs = batch["x"].size(0)
        run += float(loss.detach()) * bs
        n += bs
    return {"loss": run / max(n, 1), "lr": lr, "alpha": float(F.softplus(model.filter.alpha).detach())}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seq_len = 6000

    picker, _, _ = load_model(args.checkpoint, device)
    for p in picker.parameters():
        p.requires_grad = False
    filt = WaveformResidualFilter(hidden=args.filter_hidden, layers=args.filter_layers).to(device)
    model = FilteredC2F(picker, filt).to(device)
    n_train = sum(p.numel() for p in filt.parameters())
    print(f"[filter] trainable={n_train:,} picker frozen from {args.checkpoint}", flush=True)

    train_ds = STEADPickingDataset(
        "train",
        seq_len=seq_len,
        max_event_traces=80000,
        max_noise_traces=20000,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
        augment=True,
        fresh_augment=True,
        aug_shift_prob=0.5,
        aug_shift_p_range=(1.0, 50.0),
        aug_channel_drop_prob=0.1,
        aug_gauss_prob=0.3,
        aug_noise_snr_db=(5.0, 20.0),
        aug_second_event_prob=0.2,
        aug_real_noise_prob=args.aug_real_noise_prob,
        aug_real_noise_snr_db=(args.aug_real_noise_snr_min, args.aug_real_noise_snr_max),
        aug_response_prob=0.2,
        aug_gap_prob=0.1,
        aug_phase_absent_prob=0.2,
        aug_lowfreq_prob=0.2,
        aug_reverb_prob=0.1,
        load_p_status=True,
    )
    val_ds = STEADPickingDataset(
        "val",
        seq_len=seq_len,
        max_event_traces=args.max_val // 2,
        max_noise_traces=args.max_val // 2,
        label_sigma_sec=args.label_sigma_sec,
        seed=args.seed,
    )
    sampler = RandomSampler(
        train_ds, replacement=True, num_samples=args.samples_per_epoch
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    opt = torch.optim.AdamW(filt.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = {"pick_focus": -1.0}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, opt, device, args, seq_len, ep)
        # evaluate needs model.eval — picker already frozen
        model.eval()
        val = evaluate(
            model, val_loader, device,
            seq_len=seq_len,
            pick_th=args.pick_threshold,
            tol_sec=args.pick_tolerance_sec,
        )
        row = {"epoch": ep, **tr, "val": val, "sec": time.time() - t0}
        history.append(row)
        print(
            f"ep{ep:02d} lr={tr['lr']:.2e} loss={tr['loss']:.4f} α={tr['alpha']:.3f} "
            f"val P={val['p_f1']:.4f} S={val['s_f1']:.4f} pick_focus={val['pick_focus']:.4f} "
            f"({row['sec']:.0f}s)",
            flush=True,
        )
        ckpt = {
            "epoch": ep,
            "filter_state_dict": filt.state_dict(),
            "picker_checkpoint": str(args.checkpoint),
            "args": vars(args),
            "val": val,
            "best": best,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val["pick_focus"] > best["pick_focus"]:
            best = {"pick_focus": val["pick_focus"], "epoch": ep, **{k: val[k] for k in val}}
            ckpt["best"] = best
            torch.save(ckpt, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps({"best": best, "n_filter_params": n_train}, indent=2)
    )

    if not args.skip_snr_eval:
        # quick SNR eval via wrapping eval_snr_curve run_c2f-compatible path
        print("[snr] evaluating filtered vs bare imp_RSP…", flush=True)
        from tools.eval_snr_curve import load_sets
        from tools.eval_shift_invariance import run_c2f
        from tools.sweep_seisbench_threshold import metrics_at
        from hnf.robust_aug import add_real_noise

        # Monkey: run filtered batches
        @torch.no_grad()
        def run_filtered(waves, bs, det_th):
            dets, ps, ss = [], [], []
            T = 6000
            for a in range(0, len(waves), bs):
                x = torch.from_numpy(waves[a : a + bs]).to(device)
                x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)
                x = x.transpose(1, 2).contiguous()  # (B,T,C)
                t = torch.linspace(0.0, 60.0, T, device=device).view(1, T, 1).expand(x.size(0), -1, -1)
                out = model(x, t)
                det = torch.sigmoid(out["det"].float())
                gate = (det >= det_th).float().view(-1, 1)
                dets.append(det.cpu())
                ps.append((torch.sigmoid(out["p"].float()) * gate).half().cpu())
                ss.append((torch.sigmoid(out["s"].float()) * gate).half().cpu())
            return torch.cat(dets), torch.cat(ps), torch.cat(ss)

        # load best filter
        best_ck = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
        filt.load_state_dict(best_ck["filter_state_dict"])
        model.eval()
        bare, _, _ = load_model(args.checkpoint, device)

        ev_w, pool_w, only_w, y_ev = load_sets(2000, 500, 7, "test")
        n_ev, n_nz = len(ev_w), len(only_w)
            y = {
                "p_idx": torch.cat([y_ev["p_idx"], torch.zeros(n_nz, dtype=torch.long)]),
                "s_idx": torch.cat([y_ev["s_idx"], torch.zeros(n_nz, dtype=torch.long)]),
                "p_valid": torch.cat([torch.ones(n_ev), torch.zeros(n_nz)]).bool(),
                "s_valid": torch.cat([torch.ones(n_ev), torch.zeros(n_nz)]).bool(),
                "is_event": torch.cat([torch.ones(n_ev), torch.zeros(n_nz)]).bool(),
            }
            snr_report = {"n_events": n_ev, "n_noise": n_nz, "snrs": {}}
            for snr in ["clean", "0", "-5"]:
                if snr == "clean":
                    w_ev = ev_w.astype(np.float32)
                else:
                    w_ev = np.stack(
                        [add_real_noise(ev_w[i], int(y_ev["p_idx"][i]), pool_w[i], float(snr)) for i in range(n_ev)]
                    ).astype(np.float32)
                waves = np.concatenate([w_ev, only_w.astype(np.float32)], 0)
                row = {}
                for name, runner in [
                    ("filtered", lambda w: run_filtered(w, 64, 0.5)),
                    ("imp_RSP", lambda w: run_c2f(bare, w, device, 64, 0.5)),
                ]:
                    det, p, s = runner(waves)
                    m = metrics_at(det, p, s, y, "eqt", 0.3, 0.5, 50, device)
                    row[name] = {
                        "p_f1": m["p_f1"], "p_precision": m["p_precision"], "p_recall": m["p_recall"],
                        "s_f1": m["s_f1"], "s_precision": m["s_precision"], "s_recall": m["s_recall"],
                    }
            snr_report["snrs"][snr] = row
            print(
                f"  snr={snr}: filtered P F1={row['filtered']['p_f1']:.4f} "
                f"imp_RSP={row['imp_RSP']['p_f1']:.4f} Δ={row['filtered']['p_f1']-row['imp_RSP']['p_f1']:+.4f}",
                flush=True,
            )
        (args.output_dir / "snr_pilot.json").write_text(json.dumps(snr_report, indent=2))
        # copy summary to results
        out = Path("results/phase2/2_4_input_filter")
        out.mkdir(parents=True, exist_ok=True)
        (out / "snr_pilot.json").write_text(json.dumps(snr_report, indent=2))
        delta = snr_report["snrs"]["-5"]["filtered"]["p_f1"] - snr_report["snrs"]["-5"]["imp_RSP"]["p_f1"]
        clean_d = snr_report["snrs"]["clean"]["filtered"]["p_f1"] - snr_report["snrs"]["clean"]["imp_RSP"]["p_f1"]
        lines = [
            "# Input filter → frozen c2f pilot",
            "",
            f"- Best val pick_focus={best['pick_focus']:.4f} (ep {best.get('epoch')})",
            f"- Test SNR −5 dB P F1 Δ vs imp_RSP: **{delta:+.4f}**",
            f"- Test clean P F1 Δ: **{clean_d:+.4f}**",
            "",
            "Pilot only (3 ep / 40k samples). Not a Phase-B success claim.",
        ]
        (out / "PILOT_CONCLUSION.md").write_text("\n".join(lines) + "\n")

    print(f"[done] → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()