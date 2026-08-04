# -*- coding: utf-8 -*-
"""Differentiable Kernel Bank — evolving Huygens meta-kernel dictionary.

See ``docs/DESIGN_DIFFERENTIABLE_KERNEL_BANK.md``.

Phase-0 (this module): soft assignment, diversity / entropy losses, EMA pairwise
distance, soft merge gates, three-phase schedule hooks, top-M sparsified forward
reusing ``HuygensKernel.forward_apply``.

Discrete split / merge-with-rollback are API stubs for Phase-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnf.kernel import HuygensKernel
from hnf.layers import build_huygens_kernel


def _pairwise_euclid(x: torch.Tensor) -> torch.Tensor:
    """x: (N, D) → (N, N) Euclidean distances."""
    return torch.cdist(x, x, p=2)


@dataclass
class KernelBankScheduleState:
    """Snapshot of the three-phase training schedule."""

    phase: str  # differentiate | merge | lock
    progress: float
    merge_rate: float
    diversity_weight: float
    entropy_weight: float
    role_anchor_weight: float
    allow_discrete_merge: bool
    allow_slow_drift: bool


@dataclass
class KernelBankSchedule:
    """Epoch-fraction schedule matching the design doc."""

    differentiate_until: float = 0.20
    merge_until: float = 0.70
    diversity_weight: float = 0.05
    entropy_weight: float = 0.02
    role_anchor_weight: float = 0.05
    merge_rate0: float = 1.0
    merge_decay: float = 4.0  # merge_rate = merge_rate0 * exp(-decay * progress)

    def state(self, epoch: int, total_epochs: int) -> KernelBankScheduleState:
        total = max(int(total_epochs), 1)
        p = float(epoch) / float(total)
        if p < self.differentiate_until:
            return KernelBankScheduleState(
                phase="differentiate",
                progress=p,
                merge_rate=0.0,
                diversity_weight=self.diversity_weight,
                entropy_weight=self.entropy_weight,
                role_anchor_weight=0.0,
                allow_discrete_merge=False,
                allow_slow_drift=False,
            )
        if p < self.merge_until:
            return KernelBankScheduleState(
                phase="merge",
                progress=p,
                merge_rate=self.merge_rate0 * float(
                    torch.exp(torch.tensor(-self.merge_decay * p)).item()
                ),
                diversity_weight=0.5 * self.diversity_weight,
                entropy_weight=0.5 * self.entropy_weight,
                role_anchor_weight=0.0,
                allow_discrete_merge=True,
                allow_slow_drift=False,
            )
        return KernelBankScheduleState(
            phase="lock",
            progress=p,
            merge_rate=0.0,
            diversity_weight=0.0,
            entropy_weight=0.0,
            role_anchor_weight=self.role_anchor_weight,
            allow_discrete_merge=False,
            allow_slow_drift=True,
        )


class DifferentiableKernelBank(nn.Module):
    """Bank of ``n_kernels`` Huygens kernels with soft routing and merge gates.

    Parameters
    ----------
    n_kernels:
        Initial dictionary size N (identical init; drift during training).
    top_m:
        If >0, only the top-M assignment weights participate in the forward mix
        (complexity ~×M). If 0, use all alive kernels.
    """

    def __init__(
        self,
        n_kernels: int = 8,
        *,
        gamma: float = 1.0,
        omega: float = 1.0,
        wave_speed: float = 6.0,
        causal: bool = True,
        distance_mode: str = "time",
        local_window_sec: Optional[float] = 15.0,
        sparse_band: bool = True,
        principle: str = "huygens_fresnel",
        obliquity_scale: float = 1.0,
        obliquity_mix: float = 0.0,
        learnable_kernel_params: bool = True,
        top_m: int = 4,
        assign_hidden: int = 32,
        merge_distance_threshold: float = 0.35,
        merge_gate_temperature: float = 10.0,
        ema_momentum: float = 0.95,
        param_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        if n_kernels < 1:
            raise ValueError("n_kernels must be >= 1")
        self.n_kernels = int(n_kernels)
        self.top_m = int(top_m)
        self.merge_distance_threshold = float(merge_distance_threshold)
        self.merge_gate_temperature = float(merge_gate_temperature)
        self.ema_momentum = float(ema_momentum)
        self.param_scale = tuple(float(x) for x in param_scale)

        # Small init jitter so gradients can break symmetry.
        self.kernels = nn.ModuleList()
        for k in range(self.n_kernels):
            jitter = 1.0 + 0.02 * (k - 0.5 * (self.n_kernels - 1)) / max(self.n_kernels, 1)
            self.kernels.append(
                build_huygens_kernel(
                    gamma=gamma * jitter,
                    omega=omega / jitter,
                    wave_speed=wave_speed,
                    learnable_kernel_params=learnable_kernel_params,
                    distance_mode=distance_mode,
                    local_window_sec=local_window_sec,
                    sparse_band=sparse_band,
                    principle=principle,
                    obliquity_scale=obliquity_scale,
                    obliquity_mix=obliquity_mix,
                    causal=causal,
                )
            )

        # Soft alive mass in (0,1]; hard prune by setting near 0.
        self.alive_logit = nn.Parameter(torch.zeros(self.n_kernels))
        # Content-adaptive assignment over channel-pooled features.
        self.assign_mlp = nn.Sequential(
            nn.Linear(1, assign_hidden),
            nn.GELU(),
            nn.Linear(assign_hidden, self.n_kernels),
        )
        # Optional global mix bias (used when assignment features are absent).
        self.mix_bias = nn.Parameter(torch.zeros(self.n_kernels))

        # Role anchors (filled in lock phase); (N, 3) γ/ω/c.
        self.register_buffer("role_anchor", torch.zeros(self.n_kernels, 3))
        self.register_buffer("role_anchor_valid", torch.zeros(self.n_kernels))
        # EMA pairwise distance (N, N).
        self.register_buffer("ema_pairwise_dist", torch.zeros(self.n_kernels, self.n_kernels))
        self.register_buffer("ema_initialized", torch.zeros((), dtype=torch.bool))

        # Book-keeping for discrete merge rollback (Phase-1).
        self._merge_snapshot: dict | None = None
        self._schedule = KernelBankSchedule()

    # ------------------------------------------------------------------ params
    def alive_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.alive_logit)

    def parameter_matrix(self) -> torch.Tensor:
        """Return (N, 3) effective (γ, ω, c), scaled for distance geometry."""
        rows = []
        sx, sy, sz = self.param_scale
        for k in self.kernels:
            assert isinstance(k, HuygensKernel)
            rows.append(
                torch.stack(
                    [
                        k.effective_gamma() * sx,
                        k.effective_omega() * sy,
                        k.effective_wave_speed() * sz,
                    ]
                )
            )
        return torch.stack(rows, dim=0)

    def unscaled_parameter_matrix(self) -> torch.Tensor:
        rows = []
        for k in self.kernels:
            assert isinstance(k, HuygensKernel)
            rows.append(
                torch.stack(
                    [k.effective_gamma(), k.effective_omega(), k.effective_wave_speed()]
                )
            )
        return torch.stack(rows, dim=0)

    def update_ema_distances(self) -> torch.Tensor:
        """Update EMA pairwise distances; return current EMA matrix."""
        with torch.no_grad():
            d = _pairwise_euclid(self.parameter_matrix().detach())
            if not bool(self.ema_initialized.item()):
                self.ema_pairwise_dist.copy_(d)
                self.ema_initialized.fill_(True)
            else:
                m = self.ema_momentum
                self.ema_pairwise_dist.mul_(m).add_(d, alpha=1.0 - m)
        return self.ema_pairwise_dist

    def soft_merge_gates(self) -> torch.Tensor:
        """(N, N) gates in (0,1); high ⇒ kernels should act as one."""
        dist = self.update_ema_distances()
        # Larger when closer than threshold.
        logits = self.merge_gate_temperature * (self.merge_distance_threshold - dist)
        gates = torch.sigmoid(logits)
        eye = torch.eye(self.n_kernels, device=gates.device, dtype=gates.dtype)
        return gates * (1.0 - eye)

    # ----------------------------------------------------------- assignment
    def assignment_logits(
        self,
        h_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits (..., N). If h_real is (B,T,D), pool → (B,1) energy cue."""
        if h_real is None:
            return self.mix_bias.view(*([1] * 0), self.n_kernels)
        # Local amplitude energy as cheap content cue (B,T,1).
        energy = h_real.pow(2).mean(dim=-1, keepdim=True)
        return self.assign_mlp(energy) + self.mix_bias

    def mix_weights(
        self,
        h_real: Optional[torch.Tensor] = None,
        *,
        top_m: Optional[int] = None,
    ) -> torch.Tensor:
        """Softmax mix with alive mask; optional top-M sparsification.

        Returns weights broadcastable to multiply kernel outputs:
        - no h: (N,)
        - with h (B,T,D): (B,T,N)
        """
        logits = self.assignment_logits(h_real)
        alive = self.alive_probs()
        # Mask dead kernels.
        mask_logit = torch.log(alive.clamp_min(1e-6))
        if logits.dim() == 1:
            logits = logits + mask_logit
        else:
            logits = logits + mask_logit.view(*([1] * (logits.dim() - 1)), -1)

        m = self.top_m if top_m is None else int(top_m)
        if m > 0 and m < self.n_kernels:
            topv, topi = torch.topk(logits, k=m, dim=-1)
            sparse = torch.full_like(logits, -1e9)
            sparse.scatter_(-1, topi, topv)
            logits = sparse
        w = torch.softmax(logits, dim=-1)
        return w

    # -------------------------------------------------------------- forward
    def forward_apply(
        self,
        h_complex: torch.Tensor,
        h_real: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
        *,
        top_m: Optional[int] = None,
    ) -> torch.Tensor:
        """Mixture of bank kernels applied to complex field ``h_complex``."""
        w = self.mix_weights(h_real, top_m=top_m)  # (B,T,N) or (N,)
        # Soft-merge: blend sibling weights when gates are high (cheap coupling).
        gates = self.soft_merge_gates()  # (N,N)
        if w.dim() == 1:
            # w' = normalize(w + gates @ w)
            w = w + gates.matmul(w)
            w = w / w.sum().clamp_min(1e-6)
            out = None
            for k, kernel in enumerate(self.kernels):
                y = kernel.forward_apply(h_complex, h_real, t=t, rho=rho)
                out = y * w[k] if out is None else out + y * w[k]
            return out

        # (B,T,N): merge coupling per time step would be expensive; use mean gate
        # pull toward averaged sibling mass.
        sibling = torch.einsum("ij,btj->bti", gates, w)
        w = w + sibling
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = torch.zeros_like(h_complex)
        # Only touch kernels with non-negligible mass somewhere in the batch.
        mass = w.mean(dim=(0, 1))  # (N,)
        for k, kernel in enumerate(self.kernels):
            if float(mass[k].detach()) < 1e-4:
                continue
            y = kernel.forward_apply(h_complex, h_real, t=t, rho=rho)
            wk = w[..., k].unsqueeze(-1)  # (B,T,1)
            out = out + y * wk
        return out

    # ---------------------------------------------------------------- losses
    def diversity_loss(self) -> torch.Tensor:
        """Encourage surviving kernels to spread in (γ,ω,c) space."""
        alive = self.alive_probs()
        theta = self.parameter_matrix()
        d = _pairwise_euclid(theta)
        eye = torch.eye(self.n_kernels, device=d.device, dtype=d.dtype)
        pair_w = alive.unsqueeze(1) * alive.unsqueeze(0) * (1.0 - eye)
        # Maximize distance ≡ minimize negative mean distance.
        mean_d = (d * pair_w).sum() / pair_w.sum().clamp_min(1e-6)
        return -mean_d

    def assignment_entropy_loss(self, h_real: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encourage non-collapsed soft assignments (maximize entropy → minimize -H)."""
        w = self.mix_weights(h_real, top_m=0)  # full softmax for entropy
        if w.dim() == 1:
            ent = -(w * (w.clamp_min(1e-8).log())).sum()
        else:
            ent = -(w * (w.clamp_min(1e-8).log())).sum(dim=-1).mean()
        return -ent

    def role_anchor_loss(self) -> torch.Tensor:
        """Pull kernels toward locked role centers (lock phase)."""
        valid = self.role_anchor_valid > 0.5
        if not bool(valid.any()):
            return self.parameter_matrix().sum() * 0.0
        theta = self.unscaled_parameter_matrix()
        diff = (theta - self.role_anchor).pow(2).sum(dim=-1)
        alive = self.alive_probs()
        w = alive * valid.float()
        return (diff * w).sum() / w.sum().clamp_min(1e-6)

    def physics_prior_loss(self) -> torch.Tensor:
        losses = [k.physics_prior_loss() for k in self.kernels]
        alive = self.alive_probs()
        stacked = torch.stack(losses)
        return (stacked * alive).sum() / alive.sum().clamp_min(1e-6)

    def bank_regularizers(
        self,
        h_real: Optional[torch.Tensor],
        schedule: KernelBankScheduleState,
    ) -> dict[str, torch.Tensor]:
        zero = self.parameter_matrix().sum() * 0.0
        out = {
            "diversity": self.diversity_loss() * schedule.diversity_weight
            if schedule.diversity_weight > 0
            else zero,
            "entropy": self.assignment_entropy_loss(h_real) * schedule.entropy_weight
            if schedule.entropy_weight > 0
            else zero,
            "role_anchor": self.role_anchor_loss() * schedule.role_anchor_weight
            if schedule.role_anchor_weight > 0
            else zero,
        }
        out["total"] = out["diversity"] + out["entropy"] + out["role_anchor"]
        return out

    # -------------------------------------------------------------- roles
    @torch.no_grad()
    def capture_role_anchors(self) -> None:
        """Snapshot current effective params as role centers (call at lock start)."""
        self.role_anchor.copy_(self.unscaled_parameter_matrix())
        self.role_anchor_valid.copy_((self.alive_probs() >= 0.5).float())

    @torch.no_grad()
    def summarize(self) -> list[dict[str, float]]:
        theta = self.unscaled_parameter_matrix().detach().cpu()
        alive = self.alive_probs().detach().cpu()
        rows = []
        for i in range(self.n_kernels):
            rows.append(
                {
                    "id": i,
                    "alive": float(alive[i]),
                    "gamma": float(theta[i, 0]),
                    "omega": float(theta[i, 1]),
                    "wave_speed": float(theta[i, 2]),
                }
            )
        return rows

    # ------------------------------------------- discrete merge/split stubs
    @torch.no_grad()
    def propose_merge_pairs(
        self,
        overlap: Optional[torch.Tensor] = None,
        max_pairs: int = 1,
    ) -> list[tuple[int, int]]:
        """Return candidate (i,j) pairs with i<j for discrete merge."""
        dist = self.update_ema_distances()
        alive = self.alive_probs()
        pairs = []
        for i in range(self.n_kernels):
            if float(alive[i]) < 0.5:
                continue
            for j in range(i + 1, self.n_kernels):
                if float(alive[j]) < 0.5:
                    continue
                if float(dist[i, j]) > self.merge_distance_threshold:
                    continue
                if overlap is not None and float(overlap[i, j]) < 0.5:
                    continue
                pairs.append((i, j, float(dist[i, j])))
        pairs.sort(key=lambda x: x[2])
        return [(i, j) for i, j, _ in pairs[:max_pairs]]

    @torch.no_grad()
    def apply_soft_merge(self, i: int, j: int) -> None:
        """Blend params of j into i and suppress j (Phase-1 primitive)."""
        self._merge_snapshot = {
            "i": i,
            "j": j,
            "state_i": {k: v.detach().clone() for k, v in self.kernels[i].state_dict().items()},
            "state_j": {k: v.detach().clone() for k, v in self.kernels[j].state_dict().items()},
            "alive_logit": self.alive_logit.detach().clone(),
        }
        ki, kj = self.kernels[i], self.kernels[j]
        for name in ("gamma", "omega", "wave_speed", "c_log_scale", "obliquity_scale"):
            if not hasattr(ki, name) or not hasattr(kj, name):
                continue
            pi, pj = getattr(ki, name), getattr(kj, name)
            if isinstance(pi, nn.Parameter) and isinstance(pj, nn.Parameter):
                pi.data.mul_(0.5).add_(pj.data, alpha=0.5)
        # Kill j softly.
        self.alive_logit.data[j] = -6.0

    @torch.no_grad()
    def rollback_last_merge(self) -> bool:
        if not self._merge_snapshot:
            return False
        snap = self._merge_snapshot
        self.kernels[snap["i"]].load_state_dict(snap["state_i"], strict=False)
        self.kernels[snap["j"]].load_state_dict(snap["state_j"], strict=False)
        self.alive_logit.copy_(snap["alive_logit"])
        self._merge_snapshot = None
        return True

    @torch.no_grad()
    def split_kernel(self, k: int) -> Optional[int]:
        """Clone kernel ``k`` into a dead slot and push γ/ω apart. Returns new idx."""
        alive = self.alive_probs()
        dead = (alive < 0.2).nonzero(as_tuple=False).view(-1)
        if dead.numel() == 0:
            return None
        j = int(dead[0].item())
        self.kernels[j].load_state_dict(self.kernels[k].state_dict(), strict=False)
        # Push apart in γ/ω.
        for name, scale in (("gamma", 1.15), ("omega", 1.15)):
            pk = getattr(self.kernels[k], name)
            pj = getattr(self.kernels[j], name)
            if isinstance(pk, nn.Parameter):
                pk.data.mul_(scale)
            if isinstance(pj, nn.Parameter):
                pj.data.mul_(1.0 / scale)
        self.alive_logit.data[j] = 0.0
        return j

    def schedule_state(self, epoch: int, total_epochs: int) -> KernelBankScheduleState:
        return self._schedule.state(epoch, total_epochs)


class KernelBankWaveBlock(nn.Module):
    """Drop-in-ish WaveBlock that propagates via a DifferentiableKernelBank."""

    def __init__(
        self,
        dim: int,
        n_kernels: int = 8,
        *,
        gamma: float = 0.5,
        omega: float = 0.3,
        wave_speed: float = 6.0,
        local_window_sec: Optional[float] = 15.0,
        sparse_band: bool = True,
        principle: str = "huygens_fresnel",
        obliquity_scale: float = 1.0,
        top_m: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bank = DifferentiableKernelBank(
            n_kernels,
            gamma=gamma,
            omega=omega,
            wave_speed=wave_speed,
            local_window_sec=local_window_sec,
            sparse_band=sparse_band,
            principle=principle,
            obliquity_scale=obliquity_scale,
            top_m=top_m,
        )
        self.proj_real = nn.Linear(dim, dim, bias=False)
        self.proj_imag = nn.Linear(dim, dim, bias=False)
        self.norm_real = nn.LayerNorm(dim)
        self.norm_imag = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        rho: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = "cuda" if h_real.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            h_r = h_real.float()
            h_i = h_imag.float()
            t_f = t.float() if t is not None else None
            rho_f = rho.float() if rho is not None else None
            h_c = torch.complex(h_r, h_i)
            out_c = self.bank.forward_apply(h_c, h_r, t=t_f, rho=rho_f)
            out_real = self.dropout(self.proj_real(out_c.real))
            out_imag = self.dropout(self.proj_imag(out_c.imag))
            h_real = self.norm_real(h_r + out_real)
            h_imag = self.norm_imag(h_i + out_imag)
        return h_real, h_imag
