#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage training for the foveated active-perception engine.

Stage 1 — behavior cloning on Scheduler.policy_mlp (expert gaze = GT P/S peaks).
Stage 2 — joint fine-tune: pick BCE + causal consistency + gaze-efficiency penalty.

Default backbone: frozen run28 STEAD picking model inside FoveaProcessor (800-sample windows).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hnf.foveated import (
    FoveatedEngine,
    FoveaProcessor,
    PeripheralScanner,
    Scheduler,
)
from hnf.foveated.training import FoveatedTrainConfig, stage1_behavior_cloning_loss, stage2_joint_loss
from hnf.picking_metrics import (
    EvalAccumulator,
    apply_p_before_s_constraint,
    det_pred_from_logits,
    finalize_metrics,
    tolerance_bins,
    update_picking_counts,
)
from hnf.stead_picking_dataset import STEADPickingDataset
from tools.analyze_stead_picking import load_model
from tools.train_stead_picking import move_batch_to_device, set_seed


DEFAULT_CKPT = "outputs/run28/28_ms_fresnel_phys_20ep/best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train foveated engine (BC + joint)")
    p.add_argument("--output-dir", default="outputs/foveated/stage1_run28")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--seq-len", type=int, default=6000)
    p.add_argument("--max-gazes", type=int, default=8)
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--stage2-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr-stage1", type=float, default=3e-3)
    p.add_argument("--lr-stage2", type=float, default=1e-3)
    p.add_argument("--max-event-train", type=int, default=4000)
    p.add_argument("--max-noise-train", type=int, default=0, help="Noise traces (0 recommended for foveated BC)")
    p.add_argument("--max-val", type=int, default=400)
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--unfreeze-backbone", action="store_true", help="Allow full fovea backbone updates in stage 2")
    p.add_argument(
        "--unfreeze-heads",
        action="store_true",
        help="Unfreeze P/S heads only (dangerous with shift_downsample; off by default)",
    )
    p.add_argument("--scanner", choices=["energy", "sparse_huygens"], default="energy")
    p.add_argument("--pick-threshold", type=float, default=0.3)
    p.add_argument("--pick-tolerance-sec", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="", help="Resume foveated checkpoint (stage2)")
    return p.parse_args()


def collate_foveated(batch: list[dict]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in ("x", "t", "det", "p_idx", "s_idx", "p_valid", "s_valid", "p_target", "s_target"):
        out[k] = torch.stack([b[k] for b in batch])
    # Engine expects (B, 3, T)
    out["wave_b3t"] = out["x"].transpose(1, 2).contiguous()
    return out


def build_engine(args: argparse.Namespace, device: torch.device) -> FoveatedEngine:
    picking_model, _ckpt_args = load_model(Path(args.checkpoint), device)
    if args.freeze_backbone and not args.unfreeze_backbone:
        for p in picking_model.parameters():
            p.requires_grad_(False)
    fovea = FoveaProcessor(
        picking_model=picking_model,
        seq_len=args.seq_len,
        default_window_size=800,
        sample_rate_hz=(args.seq_len - 1) / 60.0,
    )
    scanner = PeripheralScanner(seq_len=args.seq_len, detector=args.scanner)
    scheduler = Scheduler(seq_len=args.seq_len, sample_rate_hz=fovea.sample_rate_hz)
    engine = FoveatedEngine(
        scanner=scanner,
        fovea=fovea,
        scheduler=scheduler,
        seq_len=args.seq_len,
        max_gazes=args.max_gazes,
    ).to(device)
    return engine


def scheduler_bc_features(
    heatmap: torch.Tensor,
    cover: torch.Tensor,
    tip_unc: torch.Tensor,
    tip_time_norm: torch.Tensor,
) -> torch.Tensor:
    """(B, 4) features for behavior cloning."""
    eps = 1e-6
    heat_peak = heatmap.amax(dim=-1)
    uncovered = heatmap * (1.0 - cover)
    uncovered_mass = uncovered.sum(dim=-1) / (heatmap.sum(dim=-1) + eps)
    return torch.stack([heat_peak, uncovered_mass, tip_unc, tip_time_norm], dim=-1)


def stage1_bc_batch(
    engine: FoveatedEngine,
    batch: dict[str, torch.Tensor],
    seq_len: int,
    max_gazes: int,
) -> torch.Tensor:
    wave = batch["wave_b3t"]
    p_idx = batch["p_idx"]
    s_idx = batch["s_idx"]
    p_valid = batch["p_valid"]
    s_valid = batch["s_valid"]

    with torch.no_grad():
        heatmap, _ = engine.scanner(wave, return_candidates=True)

    b = wave.size(0)
    device = wave.device
    cover = torch.zeros(b, seq_len, device=device)
    tip_unc = torch.zeros(b, device=device)
    tip_time = torch.zeros(b, device=device)
    loss = torch.zeros((), device=device)
    n_steps = 0

    for step in range(max_gazes):
        feats = scheduler_bc_features(heatmap, cover, tip_unc, tip_time)
        expert = torch.where(
            (step % 2 == 0) & (p_valid > 0.5),
            p_idx.float(),
            torch.where(s_valid > 0.5, s_idx.float(), p_idx.float()),
        )
        # Fall back to available phase when one is missing.
        missing_p = (step % 2 == 0) & (p_valid <= 0.5) & (s_valid > 0.5)
        expert = torch.where(missing_p, s_idx.float(), expert)
        valid = ((step % 2 == 0) & (p_valid > 0.5)) | ((step % 2 == 1) & (s_valid > 0.5))
        valid = valid | (p_valid <= 0.5) & (s_valid > 0.5)
        if not bool(valid.any()):
            break
        l = stage1_behavior_cloning_loss(engine, feats, expert, seq_len=seq_len)
        loss = loss + l
        n_steps += 1
        # Simulate coverage update toward expert gaze.
        for bi in range(b):
            if not bool(valid[bi]):
                continue
            c = int(expert[bi].item())
            r = engine.scheduler.cover_radius
            lo = max(0, c - r)
            hi = min(seq_len, c + r)
            cover[bi, lo:hi] = 1.0
            tip_time[bi] = expert[bi] / max(seq_len - 1, 1)
    if n_steps == 0:
        return loss.detach()  # caller skips backward
    return loss / n_steps


@torch.no_grad()
def eval_foveated(
    engine: FoveatedEngine,
    loader: DataLoader,
    *,
    seq_len: int,
    pick_threshold: float,
    tol_bins: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    engine.eval()
    acc = EvalAccumulator()
    n_gazes_sum = 0.0
    n_samples = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        wave = batch["wave_b3t"]
        out = engine(wave)
        det_pred = torch.ones(wave.size(0), dtype=torch.bool, device=device)
        p_prob, s_prob = apply_p_before_s_constraint(out.p_prob, out.s_prob, pick_threshold)
        valid = batch["p_valid"] > 0.5
        update_picking_counts(
            acc.p,
            p_prob,
            det_pred,
            batch["det"] > 0.5,
            valid,
            batch["p_idx"],
            pick_threshold,
            tol_bins,
            seq_len,
        )
        valid_s = batch["s_valid"] > 0.5
        update_picking_counts(
            acc.s,
            s_prob,
            det_pred,
            batch["det"] > 0.5,
            valid_s,
            batch["s_idx"],
            pick_threshold,
            tol_bins,
            seq_len,
        )
        n_gazes_sum += float(out.n_gazes.float().sum().item())
        n_samples += wave.size(0)
    metrics = finalize_metrics(acc)
    metrics["n_gazes_mean"] = n_gazes_sum / max(n_samples, 1)
    return metrics


def save_checkpoint(
    path: Path,
    engine: FoveatedEngine,
    args: argparse.Namespace,
    stage: str,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "args": vars(args),
            "state_dict": engine.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def train_stage1(
    engine: FoveatedEngine,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    for p in engine.parameters():
        p.requires_grad_(False)
    for p in engine.scheduler.policy_mlp.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(engine.scheduler.policy_mlp.parameters(), lr=args.lr_stage1)
    history: list[dict] = []
    for ep in range(1, args.stage1_epochs + 1):
        engine.train()
        losses = []
        for batch in tqdm(loader, desc=f"stage1 ep{ep}", leave=False):
            batch = move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = stage1_bc_batch(engine, batch, args.seq_len, args.max_gazes)
            if not loss.requires_grad:
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": ep, "stage": 1, "loss_bc": sum(losses) / max(len(losses), 1)}
        history.append(row)
        print(f"[stage1] ep{ep} bc_loss={row['loss_bc']:.4f}")
    return history


def train_stage2(
    engine: FoveatedEngine,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict]:
    """Stage-2 fine-tune.

    IMPORTANT: with ``shift_downsample`` fovea, pick locations go through argmax and
    fused maps are sparse bumps. Updating P/S heads with full-length BCE **destroys**
    the run28 zero-shot (observed: ep1 P≈0.79 → ep2 P≈0.01). Default is therefore
    to keep the backbone fully frozen and only train ``policy_mlp`` (BC already did
    most of the work; stage-2 pick loss is a weak regularizer on efficiency terms).
    """
    for p in engine.parameters():
        p.requires_grad_(False)
    for p in engine.scheduler.policy_mlp.parameters():
        p.requires_grad_(True)

    if args.unfreeze_backbone and engine.fovea.picking_model is not None:
        for p in engine.fovea.picking_model.parameters():
            p.requires_grad_(True)
    elif getattr(args, "unfreeze_heads", False) and engine.fovea.picking_model is not None:
        for name, p in engine.fovea.picking_model.named_parameters():
            if any(k in name for k in ("p_head", "s_head", "pick_head", "det_head")):
                p.requires_grad_(True)

    trainable = [p for p in engine.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters for stage-2")
    n_train = sum(p.numel() for p in trainable)
    print(f"[stage2] trainable params={n_train} (unfreeze_backbone={args.unfreeze_backbone})")

    opt = torch.optim.AdamW(trainable, lr=args.lr_stage2, weight_decay=1e-4)
    cfg = FoveatedTrainConfig(max_gazes=args.max_gazes)
    tol_bins = tolerance_bins(args.seq_len, args.pick_tolerance_sec)
    best_score = -1.0
    history: list[dict] = []
    device = torch.device(args.device)

    # Baseline before any stage-2 updates (keep if training hurts).
    base_m = eval_foveated(
        engine,
        val_loader,
        seq_len=args.seq_len,
        pick_threshold=args.pick_threshold,
        tol_bins=tol_bins,
        device=device,
    )
    print(
        f"[stage2] baseline val P={base_m['p_f1']:.3f} S={base_m['s_f1']:.3f} "
        f"gazes={base_m['n_gazes_mean']:.2f}"
    )
    save_checkpoint(out_dir / "best.pt", engine, args, "2_baseline", base_m)
    best_score = base_m["p_f1"] + base_m["s_f1"]

    for ep in range(1, args.stage2_epochs + 1):
        engine.train()
        # Keep backbone in eval-BN/dropout even if some heads are thawed.
        if engine.fovea.picking_model is not None and not args.unfreeze_backbone:
            engine.fovea.picking_model.eval()
        stats_acc: dict[str, float] = {}
        n = 0
        for batch in tqdm(train_loader, desc=f"stage2 ep{ep}", leave=False):
            batch = move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            out = engine(batch["wave_b3t"])
            # When backbone is frozen, pick BCE has no grad path — use efficiency + BC proxy.
            if args.unfreeze_backbone or getattr(args, "unfreeze_heads", False):
                loss, stats = stage2_joint_loss(
                    engine, out, batch["p_target"], batch["s_target"], cfg
                )
            else:
                l_eff = engine.gaze_efficiency_penalty(out.n_gazes, max_gazes=cfg.max_gazes)
                loss = cfg.lambda_efficiency * l_eff
                stats = {
                    "loss_total": float(loss.detach().cpu()),
                    "loss_pick": 0.0,
                    "loss_causal": 0.0,
                    "loss_efficiency": float(l_eff.detach().cpu()),
                    "n_gazes_mean": float(out.n_gazes.float().mean().detach().cpu()),
                }
            if not loss.requires_grad:
                # Efficiency alone may be constant if always max_gazes — skip.
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            for k, v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v
            n += 1
        train_row = {k: v / max(n, 1) for k, v in stats_acc.items()} if n else {"loss_total": 0.0}
        train_row["epoch"] = ep
        train_row["stage"] = 2

        val_m = eval_foveated(
            engine,
            val_loader,
            seq_len=args.seq_len,
            pick_threshold=args.pick_threshold,
            tol_bins=tol_bins,
            device=device,
        )
        train_row.update({f"val_{k}": v for k, v in val_m.items()})
        history.append(train_row)
        print(
            f"[stage2] ep{ep} loss={train_row.get('loss_total', 0):.4f} "
            f"val P={val_m['p_f1']:.3f} S={val_m['s_f1']:.3f} "
            f"gazes={val_m['n_gazes_mean']:.2f}"
        )
        score = val_m["p_f1"] + val_m["s_f1"]
        if score > best_score:
            best_score = score
            save_checkpoint(out_dir / "best.pt", engine, args, "2", val_m)
        # Early abort if clearly worse than baseline after 2 epochs.
        if ep >= 2 and score < 0.5 * (base_m["p_f1"] + base_m["s_f1"]):
            print(
                f"[stage2] ABORT: score {score:.3f} << baseline "
                f"{base_m['p_f1']+base_m['s_f1']:.3f}; keeping baseline checkpoint"
            )
            break
    return history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = STEADPickingDataset(
        "train",
        seq_len=args.seq_len,
        max_event_traces=args.max_event_train,
        max_noise_traces=args.max_noise_train,
        augment=True,
    )
    val_ds = STEADPickingDataset(
        "val",
        seq_len=args.seq_len,
        max_event_traces=args.max_val,
        max_noise_traces=max(50, args.max_val // 4),
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_foveated,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_foveated,
    )

    engine = build_engine(args, device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        engine.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Resumed foveated weights from {args.resume}")

    history: list[dict] = []
    t0 = time.time()

    if args.stage in {"1", "both"}:
        history.extend(train_stage1(engine, train_loader, args, device))
        save_checkpoint(out_dir / "stage1_last.pt", engine, args, "1", history[-1] if history else {})

    if args.stage in {"2", "both"}:
        history.extend(train_stage2(engine, train_loader, val_loader, args, out_dir))

    report = {
        "elapsed_sec": time.time() - t0,
        "args": vars(args),
        "history": history,
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
