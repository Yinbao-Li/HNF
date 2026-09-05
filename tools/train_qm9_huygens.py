#!/usr/bin/env python
"""QM9 Domain-IV wedge: M0 / M1 / H1 / H1-scr / H1-shell / H1-feat.

  python tools/train_qm9_huygens.py --device cuda --n-molecules 12000 \\
      --output-dir outputs/qm9/huygens_wedge_v2 --models M1 H1 H1-scr H1-shell H1-feat
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

from hnf.qm9_baselines import ContinuousFilterMolecule
from hnf.qm9_huygens import GeometricHuygensMolecule, NoGeomAtomMLP
from hnf.qm9_io import QM9SubsetDataset, ensure_qm9_subset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/qm9"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/qm9/huygens_wedge_v2"))
    p.add_argument("--n-molecules", type=int, default=12_000)
    p.add_argument("--target", type=str, default="gap")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--models",
        nargs="+",
        default=["M0", "M1", "H1", "H1-scr", "H1-shell", "H1-feat"],
    )
    p.add_argument(
        "--reuse-m0-m1-from",
        type=Path,
        default=None,
        help="Optional SUMMARY.json to copy M0/M1 rows instead of retraining",
    )
    return p.parse_args()


def cosine_lr(epoch, total, base_lr, warmup=2):
    if epoch <= warmup:
        return base_lr * epoch / max(warmup, 1)
    t = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def build_model(name: str, args):
    kw = dict(embed_dim=args.embed_dim, num_layers=args.num_layers)
    if name == "M0":
        return NoGeomAtomMLP(embed_dim=args.embed_dim)
    if name == "M1":
        return ContinuousFilterMolecule(**kw)
    if name == "H1":
        return GeometricHuygensMolecule(**kw, learnable_kernel_params=True)
    if name == "H1-feat":
        return GeometricHuygensMolecule(
            **kw, learnable_kernel_params=True, use_feature_distance=True
        )
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std, scramble=False, occlude_shell=None):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        pos = batch["pos"].to(device)
        z = batch["z"].to(device)
        mask = batch["mask"].to(device)
        y_raw = batch["y_raw"].numpy()
        out = model(
            pos, z, mask,
            scramble_geometry=scramble,
            occlude_shell=occlude_shell,
        )
        y_hat = out["y"] * y_std + y_mean
        preds.append(y_hat.cpu().numpy())
        trues.append(y_raw)
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(len(t))}


def train_one(model, train_loader, val_loader, *, device, epochs, lr, y_mean, y_std, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = {"mae": float("inf")}
    best_sd = None
    for epoch in range(1, epochs + 1):
        cur = cosine_lr(epoch, epochs, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            pos = batch["pos"].to(device)
            z = batch["z"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            pred = model(pos, z, mask, scramble_geometry=False)["y"]
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach().cpu()) * y.size(0)
            n += y.size(0)
        va = evaluate(model, val_loader, device, y_mean, y_std, scramble=False)
        print(
            f"  ep{epoch:02d} lr={cur:.2e} tr_mse={run/max(n,1):.4f} "
            f"val_mae={va['mae']:.5f} val_rmse={va['rmse']:.5f}",
            flush=True,
        )
        if va["mae"] < best["mae"]:
            best = {**va, "epoch": epoch}
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)
    return best


def write_report(path: Path, args, n_train, n_val, n_test, rows):
    lines = [
        f"# QM9 geometric Huygens wedge v2 ({args.target})",
        "",
        f"Subset n={args.n_molecules} | train/val/test={n_train}/{n_val}/{n_test}",
        "",
        "H1 v2 = Huygens phase/envelope × RBF amplitude (matched to M1 CFConv capacity).",
        "",
        "| Model | test MAE | test RMSE | params | notes |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        rid = r["id"]
        if rid == "H1-scr":
            lines.append(
                f"| {rid} | {r['test_mae_scramble']:.5f} (scr) | "
                f"{r['test_rmse_scramble']:.5f} | {r['n_params']} | "
                f"ΔMAE={r['delta_mae_scr_minus_clean']:+.5f} "
                f"{'PASS' if r['pass_geometry_gate'] else 'FAIL'} |"
            )
        elif rid == "H1-shell":
            s1 = r["shell1_mae"]
            s2 = r["shell2_mae"]
            clean = r["test_mae_clean"]
            lines.append(
                f"| {rid} | s1={s1:.5f} / s2={s2:.5f} | — | {r['n_params']} | "
                f"clean={clean:.5f}; "
                f"Δ1={s1-clean:+.5f} Δ2={s2-clean:+.5f} |"
            )
        else:
            lines.append(
                f"| {rid} | **{r['test_mae']:.5f}** | {r['test_rmse']:.5f} | "
                f"{r['n_params']} | val_ep={r.get('val_best', {}).get('epoch')} |"
            )
    path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache = ensure_qm9_subset(args.data_dir, n_molecules=args.n_molecules, seed=args.seed)
    train_ds = QM9SubsetDataset(cache, "train", target=args.target, seed=args.seed)
    val_ds = QM9SubsetDataset(
        cache, "val", target=args.target, seed=args.seed,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std,
    )
    test_ds = QM9SubsetDataset(
        cache, "test", target=args.target, seed=args.seed,
        y_mean=train_ds.y_mean, y_std=train_ds.y_std,
    )
    print(
        f"[qm9] target={args.target} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} y_mean={train_ds.y_mean:.5f} y_std={train_ds.y_std:.5f}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    rows = []
    h1_path = args.output_dir / "H1_best.pt"
    reused = set()
    if args.reuse_m0_m1_from and args.reuse_m0_m1_from.exists():
        prev = json.loads(args.reuse_m0_m1_from.read_text())
        for r in prev.get("rows", []):
            if r.get("id") in {"M0", "M1"} and r["id"] in args.models:
                rows.append(r)
                reused.add(r["id"])
                print(f"[reuse] {r['id']} from {args.reuse_m0_m1_from}", flush=True)

    for name in args.models:
        if name in reused:
            continue
        print(f"\n{'='*60}\n[{name}]\n{'='*60}", flush=True)

        if name == "H1-scr":
            if not h1_path.exists():
                print("[skip] H1 checkpoint missing", flush=True)
                continue
            model = GeometricHuygensMolecule(
                embed_dim=args.embed_dim, num_layers=args.num_layers,
                learnable_kernel_params=True,
            ).to(device)
            model.load_state_dict(torch.load(h1_path, map_location=device, weights_only=True))
            te = evaluate(
                model, test_loader, device, train_ds.y_mean, train_ds.y_std, scramble=True
            )
            te_clean = evaluate(
                model, test_loader, device, train_ds.y_mean, train_ds.y_std, scramble=False
            )
            row = {
                "id": name,
                "test_mae_scramble": te["mae"],
                "test_rmse_scramble": te["rmse"],
                "test_mae_clean": te_clean["mae"],
                "delta_mae_scr_minus_clean": te["mae"] - te_clean["mae"],
                "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "pass_geometry_gate": te["mae"] > te_clean["mae"],
            }
            rows.append(row)
            print(
                f"  → clean MAE={te_clean['mae']:.5f} scramble MAE={te['mae']:.5f} "
                f"Δ={row['delta_mae_scr_minus_clean']:+.5f} "
                f"gate={'PASS' if row['pass_geometry_gate'] else 'FAIL'}",
                flush=True,
            )
            (args.output_dir / f"{name}.json").write_text(json.dumps(row, indent=2))
            continue

        if name == "H1-shell":
            if not h1_path.exists():
                print("[skip] H1 checkpoint missing", flush=True)
                continue
            model = GeometricHuygensMolecule(
                embed_dim=args.embed_dim, num_layers=args.num_layers,
                learnable_kernel_params=True,
            ).to(device)
            model.load_state_dict(torch.load(h1_path, map_location=device, weights_only=True))
            te_clean = evaluate(
                model, test_loader, device, train_ds.y_mean, train_ds.y_std
            )
            te_s1 = evaluate(
                model, test_loader, device, train_ds.y_mean, train_ds.y_std, occlude_shell=1
            )
            te_s2 = evaluate(
                model, test_loader, device, train_ds.y_mean, train_ds.y_std, occlude_shell=2
            )
            row = {
                "id": name,
                "test_mae_clean": te_clean["mae"],
                "shell1_mae": te_s1["mae"],
                "shell2_mae": te_s2["mae"],
                "delta_shell1": te_s1["mae"] - te_clean["mae"],
                "delta_shell2": te_s2["mae"] - te_clean["mae"],
                "shell_edges_angstrom": [0.0, 1.8, 3.2, 5.0],
                "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "pass_locality_gate": te_s1["mae"] >= te_s2["mae"],
            }
            rows.append(row)
            print(
                f"  → clean={te_clean['mae']:.5f} shell1={te_s1['mae']:.5f} "
                f"shell2={te_s2['mae']:.5f} "
                f"gate={'PASS' if row['pass_locality_gate'] else 'WEAK'} "
                f"(expect shell1≥shell2)",
                flush=True,
            )
            (args.output_dir / f"{name}.json").write_text(json.dumps(row, indent=2))
            continue

        model = build_model(name, args).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best = train_one(
            model, train_loader, val_loader,
            device=device, epochs=args.epochs, lr=args.lr,
            y_mean=train_ds.y_mean, y_std=train_ds.y_std, seed=args.seed,
        )
        te = evaluate(model, test_loader, device, train_ds.y_mean, train_ds.y_std)
        row = {
            "id": name,
            "val_best": best,
            "test_mae": te["mae"],
            "test_rmse": te["rmse"],
            "n_params": n_params,
        }
        rows.append(row)
        print(f"  → test MAE={te['mae']:.5f} RMSE={te['rmse']:.5f} params={n_params:,}", flush=True)
        (args.output_dir / f"{name}.json").write_text(json.dumps(row, indent=2))
        if name == "H1":
            torch.save(model.state_dict(), h1_path)
        if name == "H1-feat":
            torch.save(model.state_dict(), args.output_dir / "H1_feat_best.pt")

    # Keep report order stable
    order = ["M0", "M1", "H1", "H1-scr", "H1-shell", "H1-feat"]
    rows_sorted = sorted(
        rows,
        key=lambda r: order.index(r["id"]) if r["id"] in order else 99,
    )

    summary = {
        "dataset": "QM9 subset",
        "n_molecules": args.n_molecules,
        "target": args.target,
        "target_unit_note": "gap in Hartree (raw QM9 units)",
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "h1_note": "v2: Huygens phase/envelope × RBF amplitude (CFConv-matched)",
        "rows": rows_sorted,
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    write_report(
        args.output_dir / "REPORT.md",
        args, len(train_ds), len(val_ds), len(test_ds), rows_sorted,
    )


if __name__ == "__main__":
    main()
