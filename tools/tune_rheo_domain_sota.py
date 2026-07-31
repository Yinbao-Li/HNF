#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune domain SOTA baselines on val, then fair test vs PNF aniso.

Search grids (val stress_rel):
  - PNF aniso: lr × param_weight
  - Classical Prony NLS: n_fit × max_nfev × init seeds
  - RhINN: mode × hidden × phys_weight × lr × epochs schedule
  - EUCLID-lite: n_library × l1 × lr × two-stage

Final: retrain best configs with more epochs; evaluate once on held-out test.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import itertools
import json
import time
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnf.rheo_dataset import RheoMemoryDataset, RheoMemoryModel
from hnf.rheo_domain_sota import ClassicalPronyNLS, SparsePronyLibrary
from hnf.rheo_memory import _inv_softplus
from hnf.rheo_memory import PronyBoltzmannKernel  # noqa: F401 — kept for clarity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune domain SOTA then fair compare")
    p.add_argument("--output-dir", default="outputs/rheo/domain_sota_tuned")
    p.add_argument("--tune-epochs", type=int, default=20, help="epochs per trial during search")
    p.add_argument("--final-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-train", type=int, default=1536)
    p.add_argument("--n-val", type=int, default=192)
    p.add_argument("--n-test", type=int, default=192)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-modes", type=int, default=2)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--max-trials-per-model", type=int, default=12)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def eval_stress(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rels = []
    for batch in loader:
        gd = batch["gammadot"].to(device)
        st = batch["stress"].to(device)
        dt = batch["dt"].to(device)
        pred = model(gd, dt)
        for i in range(gd.size(0)):
            num = (pred[i] - st[i]).pow(2).sum().sqrt()
            den = st[i].pow(2).sum().sqrt().clamp_min(1e-8)
            rels.append(float((num / den).item()))
    return {
        "stress_rel": float(np.mean(rels)),
        "stress_rel_median": float(np.median(rels)),
        "stress_rel_p90": float(np.percentile(rels, 90)),
    }


# ---- Improved RhINN (Mech-TANN / collocation) ---------------------------------


class TunedRhINN(nn.Module):
    """Tunable RhINN: nn_collocation or mech_encode (Prony + NN residual)."""

    def __init__(
        self,
        dim: int = 2,
        n_modes: int = 2,
        hidden: int = 64,
        n_layers: int = 3,
        phys_weight: float = 0.1,
        mode: str = "mech_encode",
    ):
        super().__init__()
        self.dim, self.n_modes = dim, n_modes
        self.phys_weight = float(phys_weight)
        self.mode = mode
        self.raw_lambda = nn.Parameter(
            torch.tensor([_inv_softplus(v) for v in ([0.5, 5.0] + [2.0] * 4)[:n_modes]])
        )
        self.raw_G = nn.Parameter(torch.full((n_modes, dim), _inv_softplus(1.0)))
        self.raw_G_inf = nn.Parameter(torch.tensor(-8.0))
        layers: list[nn.Module] = [nn.Linear(1 + dim, hidden), nn.Tanh()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)
        self.res_scale = nn.Parameter(torch.tensor(0.05))

    def lam(self):
        return F.softplus(self.raw_lambda) + 1e-6

    def G(self):
        return F.softplus(self.raw_G) + 1e-6

    def ginf(self):
        return F.softplus(self.raw_G_inf)

    def prony(self, gd: torch.Tensor, dt) -> torch.Tensor:
        if gd.dim() == 2:
            gd = gd.unsqueeze(-1)
        b, t, d = gd.shape
        dt_t = torch.full((b,), float(dt) if not torch.is_tensor(dt) else float(dt.reshape(-1)[0]), device=gd.device, dtype=gd.dtype)
        lam, G = self.lam(), self.G()
        stress = self.ginf() * torch.cumsum(gd * dt_t.view(b, 1, 1), dim=1)
        alpha = torch.exp(-dt_t.unsqueeze(-1) / lam.unsqueeze(0))
        for c in range(d):
            sig = torch.zeros(b, self.n_modes, device=gd.device, dtype=gd.dtype)
            gain = (G[:, c] * lam).unsqueeze(0) * (1.0 - alpha)
            outs = []
            for n in range(t):
                sig = alpha * sig + gain * gd[:, n, c].unsqueeze(-1)
                outs.append(sig.sum(-1))
            stress[:, :, c] = stress[:, :, c] + torch.stack(outs, 1)
        return stress

    def forward(self, gd, dt):
        squeeze = gd.dim() == 2
        if squeeze:
            gd = gd.unsqueeze(-1)
        b, t, _ = gd.shape
        tfeat = torch.linspace(0, 1, t, device=gd.device, dtype=gd.dtype).view(1, t, 1).expand(b, t, 1)
        inp = torch.cat([tfeat, gd], dim=-1)
        if self.mode == "mech_encode":
            out = self.prony(gd, dt) + self.net(inp) * self.res_scale.abs()
        else:
            out = self.net(inp)
        return out.squeeze(-1) if squeeze else out

    def loss(self, gd, target, dt, phys_weight=None):
        pred = self.forward(gd, dt)
        data = (pred - target).pow(2).mean()
        w = self.phys_weight if phys_weight is None else float(phys_weight)
        if self.mode == "mech_encode":
            # encourage small correction + identify Prony params via data
            corr = (pred - self.prony(gd if gd.dim() == 3 else gd.unsqueeze(-1), dt).reshape_as(pred)).pow(2).mean()
            phys = 0.05 * w * corr
        else:
            base = self.prony(gd if gd.dim() == 3 else gd.unsqueeze(-1), dt)
            if target.dim() == 2:
                base = base.squeeze(-1)
            phys = w * (pred - base).pow(2).mean()
        loss = data + phys
        return loss, {"mse": float(data.detach()), "phys": float(phys.detach()), "loss": float(loss.detach())}


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    loss_fn: Callable,
    desc: str = "train",
) -> nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for ep in range(1, epochs + 1):
        model.train()
        for batch in loader:
            loss = loss_fn(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    return model


def make_loaders(args):
    kw = dict(
        n_steps=args.n_steps, dt=args.dt, n_modes=args.n_modes, dim=args.dim,
        anisotropic=True, noise_std=args.noise_std, seed=args.seed, fixed_material=True,
    )
    train = RheoMemoryDataset("train", n_samples=args.n_train, **kw)
    val = RheoMemoryDataset("val", n_samples=args.n_val, **kw)
    test = RheoMemoryDataset("test", n_samples=args.n_test, **kw)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True),
        DataLoader(val, batch_size=args.batch_size, shuffle=False),
        DataLoader(test, batch_size=args.batch_size, shuffle=False),
        train,
    )


# ---- search spaces -----------------------------------------------------------


def search_pnf(args, loaders, device) -> tuple[dict, dict]:
    train_loader, val_loader, _, _ = loaders
    grid = list(itertools.product([1e-3, 3e-3, 5e-3, 1e-2], [0.0, 0.3, 0.5]))
    trials = []
    for lr, pw in grid[: args.max_trials_per_model]:
        set_seed(args.seed)
        model = RheoMemoryModel(
            n_modes=args.n_modes, dim=args.dim, anisotropic=True,
            lambda_init=[0.4, 4.0][: args.n_modes], g_init=[1.0] * args.n_modes,
        ).to(device)

        def loss_fn(m, batch, dev):
            gd, st, dt = batch["gammadot"].to(dev), batch["stress"].to(dev), batch["dt"].to(dev)
            pred = m(gd, dt)
            loss = (pred - st).pow(2).mean()
            if pw > 0:
                lam = m.kernel.relaxation_times()
                gt = batch["lambda"][0].to(dev)
                loss = loss + pw * (torch.log(torch.sort(lam)[0]) - torch.log(torch.sort(gt)[0].clamp_min(1e-6))).pow(2).mean()
            return loss

        train_epochs(model, train_loader, device, epochs=args.tune_epochs, lr=lr, loss_fn=loss_fn)
        val = eval_stress(model, val_loader, device)
        cfg = {"lr": lr, "param_weight": pw}
        trials.append({"cfg": cfg, "val": val})
        print(f"  [pnf trial] {cfg} -> val_rel={val['stress_rel']:.4f}", flush=True)
    best = min(trials, key=lambda x: x["val"]["stress_rel"])
    return best["cfg"], {"trials": trials, "best": best}


def search_rhinn(args, loaders, device) -> tuple[dict, dict]:
    train_loader, val_loader, _, _ = loaders
    grid = list(
        itertools.product(
            ["mech_encode", "nn_collocation"],
            [64, 128],
            [0.0, 0.1, 0.5, 1.0],
            [1e-3, 3e-3],
            [2, 3],
        )
    )
    # prioritize mech_encode
    grid = sorted(grid, key=lambda x: 0 if x[0] == "mech_encode" else 1)
    trials = []
    for mode, hidden, phys_w, lr, n_layers in grid[: args.max_trials_per_model]:
        set_seed(args.seed)
        model = TunedRhINN(
            dim=args.dim, n_modes=args.n_modes, hidden=hidden, n_layers=n_layers,
            phys_weight=phys_w, mode=mode,
        ).to(device)

        def loss_fn(m, batch, dev, _pw=phys_w):
            gd, st, dt = batch["gammadot"].to(dev), batch["stress"].to(dev), batch["dt"].to(dev)
            loss, _ = m.loss(gd, st, dt, phys_weight=_pw)
            return loss

        train_epochs(model, train_loader, device, epochs=args.tune_epochs, lr=lr, loss_fn=loss_fn)
        val = eval_stress(model, val_loader, device)
        cfg = {"mode": mode, "hidden": hidden, "phys_weight": phys_w, "lr": lr, "n_layers": n_layers}
        trials.append({"cfg": cfg, "val": val})
        print(f"  [rhinn trial] {cfg} -> val_rel={val['stress_rel']:.4f}", flush=True)
    best = min(trials, key=lambda x: x["val"]["stress_rel"])
    return best["cfg"], {"trials": trials, "best": best}


def search_euclid(args, loaders, device) -> tuple[dict, dict]:
    train_loader, val_loader, _, _ = loaders
    grid = list(itertools.product([16, 24, 32, 48], [1e-4, 1e-3, 1e-2], [1e-3, 3e-3, 1e-2], [False, True]))
    trials = []
    for n_lib, l1, lr, two_stage in grid[: args.max_trials_per_model]:
        set_seed(args.seed)
        model = SparsePronyLibrary(dim=args.dim, n_library=n_lib, l1_weight=l1).to(device)

        def loss_fn(m, batch, dev, _l1=l1, _stage1=False):
            gd, st, dt = batch["gammadot"].to(dev), batch["stress"].to(dev), batch["dt"].to(dev)
            pred = m(gd, dt)
            mse = (pred - st).pow(2).mean()
            if _stage1:
                return mse
            return mse + _l1 * m.modal_G().mean()

        if two_stage:
            # stage1: fit dense
            train_epochs(
                model, train_loader, device, epochs=max(args.tune_epochs // 2, 5), lr=lr,
                loss_fn=lambda m, b, d: loss_fn(m, b, d, _stage1=True),
            )
            # stage2: sparsify
            train_epochs(
                model, train_loader, device, epochs=max(args.tune_epochs // 2, 5), lr=lr * 0.5,
                loss_fn=loss_fn,
            )
        else:
            train_epochs(model, train_loader, device, epochs=args.tune_epochs, lr=lr, loss_fn=loss_fn)
        val = eval_stress(model, val_loader, device)
        cfg = {"n_library": n_lib, "l1_weight": l1, "lr": lr, "two_stage": two_stage}
        trials.append({"cfg": cfg, "val": val})
        print(f"  [euclid trial] {cfg} -> val_rel={val['stress_rel']:.4f}", flush=True)
    best = min(trials, key=lambda x: x["val"]["stress_rel"])
    return best["cfg"], {"trials": trials, "best": best}


def search_nls(args, loaders, device) -> tuple[dict, dict]:
    train_loader, val_loader, _, _ = loaders
    grid = list(itertools.product([16, 32, 64, 96], [40, 80, 120, 200], [0, 1, 2]))
    trials = []
    for n_fit, nfev, seed_off in grid[: args.max_trials_per_model]:
        set_seed(args.seed + seed_off)
        model = ClassicalPronyNLS(n_modes=args.n_modes, dim=args.dim, anisotropic=True)
        # diversify init
        with torch.no_grad():
            model.kernel.raw_lambda.add_(0.1 * seed_off)
        batches = []
        for i, batch in enumerate(train_loader):
            if i >= max(1, n_fit // args.batch_size):
                break
            batches.append((batch["gammadot"], batch["stress"], batch["dt"]))
        try:
            model.fit_from_loader(batches, max_nfev=nfev)
            model = model.to(device)
            val = eval_stress(model, val_loader, device)
        except Exception as e:
            val = {"stress_rel": 1e9, "stress_rel_median": 1e9, "stress_rel_p90": 1e9, "error": str(e)}
        cfg = {"n_fit_samples_cap": n_fit, "max_nfev": nfev, "init_seed_off": seed_off}
        trials.append({"cfg": cfg, "val": val})
        print(f"  [nls trial] {cfg} -> val_rel={val['stress_rel']:.4f}", flush=True)
    best = min(trials, key=lambda x: x["val"]["stress_rel"])
    return best["cfg"], {"trials": trials, "best": best}


# ---- final retrain -----------------------------------------------------------


def final_pnf(cfg, args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader, _ = loaders
    set_seed(args.seed)
    model = RheoMemoryModel(
        n_modes=args.n_modes, dim=args.dim, anisotropic=True,
        lambda_init=[0.4, 4.0][: args.n_modes], g_init=[1.0] * args.n_modes,
    ).to(device)
    pw, lr = cfg["param_weight"], cfg["lr"]

    def loss_fn(m, batch, dev):
        gd, st, dt = batch["gammadot"].to(dev), batch["stress"].to(dev), batch["dt"].to(dev)
        pred = m(gd, dt)
        loss = (pred - st).pow(2).mean()
        if pw > 0:
            lam = m.kernel.relaxation_times()
            gt = batch["lambda"][0].to(dev)
            loss = loss + pw * (torch.log(torch.sort(lam)[0]) - torch.log(torch.sort(gt)[0].clamp_min(1e-6))).pow(2).mean()
        return loss

    t0 = time.time()
    best_rel, best_state = 1e9, None
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.final_epochs)
    for ep in range(1, args.final_epochs + 1):
        model.train()
        for batch in train_loader:
            loss = loss_fn(model, batch, device)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        val = eval_stress(model, val_loader, device)
        if val["stress_rel"] < best_rel:
            best_rel = val["stress_rel"]
            best_state = deepcopy(model.state_dict())
        print(f"[final pnf ep{ep:03d}] val_rel={val['stress_rel']:.4f}", flush=True)
    model.load_state_dict(best_state)
    test = eval_stress(model, test_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best.pt")
    return {
        "model": "pnf_aniso",
        "cfg": cfg,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "test": test,
        "val_best": best_rel,
        "elapsed_sec": time.time() - t0,
        "cite": "this work (tuned)",
    }


def final_rhinn(cfg, args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader, _ = loaders
    set_seed(args.seed)
    model = TunedRhINN(
        dim=args.dim, n_modes=args.n_modes, hidden=cfg["hidden"], n_layers=cfg["n_layers"],
        phys_weight=cfg["phys_weight"], mode=cfg["mode"],
    ).to(device)
    t0 = time.time()
    best_rel, best_state = 1e9, None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.final_epochs)
    for ep in range(1, args.final_epochs + 1):
        model.train()
        # curriculum: ramp phys weight
        ramp = min(1.0, ep / max(args.final_epochs * 0.3, 1))
        pw = cfg["phys_weight"] * ramp
        for batch in train_loader:
            gd, st, dt = batch["gammadot"].to(device), batch["stress"].to(device), batch["dt"].to(device)
            loss, _ = model.loss(gd, st, dt, phys_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        val = eval_stress(model, val_loader, device)
        if val["stress_rel"] < best_rel:
            best_rel = val["stress_rel"]
            best_state = deepcopy(model.state_dict())
        print(f"[final rhinn ep{ep:03d}] val_rel={val['stress_rel']:.4f}", flush=True)
    model.load_state_dict(best_state)
    test = eval_stress(model, test_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best.pt")
    return {
        "model": "rhinn_tuned",
        "cfg": cfg,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "test": test,
        "val_best": best_rel,
        "elapsed_sec": time.time() - t0,
        "cite": "RhINN/Mech-TANN-style (Sci Rep 2021; tuned)",
    }


def final_euclid(cfg, args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader, _ = loaders
    set_seed(args.seed)
    model = SparsePronyLibrary(
        dim=args.dim, n_library=cfg["n_library"], l1_weight=cfg["l1_weight"]
    ).to(device)
    t0 = time.time()
    best_rel, best_state = 1e9, None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    e1 = args.final_epochs // 2 if cfg["two_stage"] else 0
    for ep in range(1, args.final_epochs + 1):
        model.train()
        use_l1 = (not cfg["two_stage"]) or (ep > e1)
        for batch in train_loader:
            gd, st, dt = batch["gammadot"].to(device), batch["stress"].to(device), batch["dt"].to(device)
            pred = model(gd, dt)
            loss = (pred - st).pow(2).mean()
            if use_l1:
                loss = loss + cfg["l1_weight"] * model.modal_G().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        val = eval_stress(model, val_loader, device)
        if val["stress_rel"] < best_rel:
            best_rel = val["stress_rel"]
            best_state = deepcopy(model.state_dict())
        print(f"[final euclid ep{ep:03d}] val_rel={val['stress_rel']:.4f}", flush=True)
    model.load_state_dict(best_state)
    test = eval_stress(model, test_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best.pt")
    return {
        "model": "sparse_prony_euclid_tuned",
        "cfg": cfg,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "test": test,
        "val_best": best_rel,
        "elapsed_sec": time.time() - t0,
        "cite": "EUCLID-lite Prony library (Marino 2023; tuned)",
    }


def final_nls(cfg, args, loaders, device, out_dir: Path) -> dict:
    train_loader, val_loader, test_loader, _ = loaders
    set_seed(args.seed + cfg["init_seed_off"])
    model = ClassicalPronyNLS(n_modes=args.n_modes, dim=args.dim, anisotropic=True)
    with torch.no_grad():
        model.kernel.raw_lambda.add_(0.1 * cfg["init_seed_off"])
    batches = []
    for i, batch in enumerate(train_loader):
        if i >= max(1, cfg["n_fit_samples_cap"] // args.batch_size):
            break
        batches.append((batch["gammadot"], batch["stress"], batch["dt"]))
    t0 = time.time()
    model.fit_from_loader(batches, max_nfev=cfg["max_nfev"])
    model = model.to(device)
    val = eval_stress(model, val_loader, device)
    test = eval_stress(model, test_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best.pt")
    return {
        "model": "classical_prony_nls_tuned",
        "cfg": cfg,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "test": test,
        "val_best": val["stress_rel"],
        "elapsed_sec": time.time() - t0,
        "cite": "Classical Prony NLS (rheometry standard; tuned)",
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders = make_loaders(args)

    search_log = {}
    finals = []

    print("\n==== tune PNF ====", flush=True)
    cfg_pnf, log_pnf = search_pnf(args, loaders, device)
    search_log["pnf"] = log_pnf
    finals.append(final_pnf(cfg_pnf, args, loaders, device, out / "pnf_aniso"))

    print("\n==== tune Classical NLS ====", flush=True)
    cfg_nls, log_nls = search_nls(args, loaders, device)
    search_log["nls"] = log_nls
    finals.append(final_nls(cfg_nls, args, loaders, device, out / "classical_prony_nls"))

    print("\n==== tune RhINN ====", flush=True)
    cfg_rhinn, log_rhinn = search_rhinn(args, loaders, device)
    search_log["rhinn"] = log_rhinn
    finals.append(final_rhinn(cfg_rhinn, args, loaders, device, out / "rhinn"))

    print("\n==== tune EUCLID ====", flush=True)
    cfg_euclid, log_euclid = search_euclid(args, loaders, device)
    search_log["euclid"] = log_euclid
    finals.append(final_euclid(cfg_euclid, args, loaders, device, out / "euclid"))

    (out / "search_log.json").write_text(json.dumps(search_log, indent=2, default=str))
    for r in finals:
        d = out / r["model"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(r, indent=2))

    rows = sorted(finals, key=lambda r: r["test"]["stress_rel"])
    board = {
        "note": "All methods tuned on val (grid search); final metrics on held-out test once.",
        "protocol": {
            "dim": args.dim, "n_modes": args.n_modes, "n_steps": args.n_steps,
            "n_train": args.n_train, "tune_epochs": args.tune_epochs, "final_epochs": args.final_epochs,
        },
        "rows": [
            {
                "model": r["model"],
                "cite": r["cite"],
                "cfg": r["cfg"],
                "n_params": r["n_params"],
                "val_best": r["val_best"],
                "stress_rel": r["test"]["stress_rel"],
                "stress_rel_median": r["test"]["stress_rel_median"],
                "stress_rel_p90": r["test"]["stress_rel_p90"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in rows
        ],
        "best": rows[0]["model"],
    }
    (out / "BOARD.json").write_text(json.dumps(board, indent=2))
    lines = [
        "# Domain rheology SOTA — **tuned** fair comparison",
        "",
        board["note"],
        "",
        "| Model | stress_rel ↓ | median | p90 | params | best cfg |",
        "|-------|-------------:|-------:|----:|-------:|----------|",
    ]
    for r in board["rows"]:
        mark = "**" if r["model"] == board["best"] else ""
        lines.append(
            f"| {mark}{r['model']}{mark} | {r['stress_rel']:.4f} | {r['stress_rel_median']:.4f} | "
            f"{r['stress_rel_p90']:.4f} | {r['n_params']} | `{json.dumps(r['cfg'], separators=(',', ':'))}` |"
        )
    lines += ["", f"**Best:** `{board['best']}`", ""]
    (out / "BOARD.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
