"""Geometric Huygens molecule model (Domain IV / QM9 wedge).

Atoms = secondary sources; pairwise Euclidean distances drive a Huygens-style
complex kernel. Static graphs → causal=False.

v2: match M1 expressivity by giving the *radial amplitude* an RBF filter net
(same capacity as CFConv) while keeping Huygens phase + 1/r² envelope on
geometry. Cutoff / no-self / shell occlusion are explicit.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomSecondarySources(nn.Module):
    """Embed atomic numbers as secondary-source amplitudes."""

    def __init__(self, embed_dim: int = 64, max_z: int = 20):
        super().__init__()
        self.embed = nn.Embedding(max_z + 1, embed_dim, padding_idx=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.embed(z.clamp(min=0))


class AtomicMediumDensity(nn.Module):
    """Per-atom medium density ρ from local embedding."""

    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class GeometricHuygensConv(nn.Module):
    """One gather step: Huygens phase/envelope × learnable RBF amplitude.

    Amplitude path mirrors CFConv (matched budget). Phase/envelope keep the
    Huygens inductive bias on true Euclidean r_ij.
    """

    def __init__(
        self,
        dim: int,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        gamma: float = 0.5,
        omega: float = 0.3,
        learnable_kernel_params: bool = True,
    ):
        super().__init__()
        self.cutoff = float(cutoff)
        self.n_rbf = int(n_rbf)
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = cutoff / max(n_rbf, 1)

        if learnable_kernel_params:
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
            self.omega = nn.Parameter(torch.tensor(float(omega)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
            self.register_buffer("omega", torch.tensor(float(omega)))

        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.proj_real = nn.Linear(dim, dim, bias=False)
        self.proj_imag = nn.Linear(dim, dim, bias=False)
        self.norm_real = nn.LayerNorm(dim)
        self.norm_imag = nn.LayerNorm(dim)

    def _eff_gamma(self) -> torch.Tensor:
        if isinstance(self.gamma, nn.Parameter):
            return F.softplus(self.gamma) + 1e-3
        return self.gamma

    def _eff_omega(self) -> torch.Tensor:
        if isinstance(self.omega, nn.Parameter):
            return F.softplus(self.omega).clamp_min(1e-3)
        return self.omega.abs().clamp_min(1e-3)

    def _pair_mask(
        self,
        d: torch.Tensor,
        atom_mask: torch.Tensor,
        occlude_shell: Optional[int] = None,
        shell_edges: tuple[float, ...] = (0.0, 1.8, 3.2, 5.0),
    ) -> torch.Tensor:
        m = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        eye = torch.eye(d.size(-1), device=d.device, dtype=torch.bool).unsqueeze(0)
        m = m & ~eye & (d <= self.cutoff)
        if occlude_shell is not None:
            # shell index 1-based over consecutive distance bins
            lo = shell_edges[occlude_shell - 1] if occlude_shell >= 1 else 0.0
            hi = shell_edges[occlude_shell] if occlude_shell < len(shell_edges) else self.cutoff
            in_shell = (d > lo) & (d <= hi)
            m = m & ~in_shell
        return m

    def forward(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
        occlude_shell: Optional[int] = None,
        use_feature_distance: bool = False,
        feat_for_distance: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Distance from coords (geometry) or latent features (H1-feat null).
        if use_feature_distance:
            x = feat_for_distance if feat_for_distance is not None else h_real
            d = torch.cdist(x, x)
        else:
            d = torch.cdist(pos, pos)

        m = self._pair_mask(d, atom_mask, occlude_shell=occlude_shell)
        rbf = torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (self.width ** 2 + 1e-8))
        amp = self.filter_net(rbf)  # (B,N,N,D)

        # Huygens radial factors on geometry (or feature) distance
        r = d.clamp_min(1e-6)
        sph = 1.0 / (r.pow(2) + 1e-6)
        env = torch.exp(-self._eff_gamma() * r.pow(2))
        phase = self._eff_omega() * r
        if rho is not None:
            rho_1d = rho.squeeze(-1)
            rho_mean = 0.5 * (rho_1d.unsqueeze(-1) + rho_1d.unsqueeze(-2))
            env = env * torch.exp(-rho_mean * r)

        scale = (sph * env).unsqueeze(-1)  # (B,N,N,1)
        cos_p = torch.cos(phase).unsqueeze(-1)
        sin_p = torch.sin(phase).unsqueeze(-1)
        w_r = amp * scale * cos_p
        w_i = amp * scale * sin_p
        w_r = w_r * m.unsqueeze(-1).float()
        w_i = w_i * m.unsqueeze(-1).float()

        # Complex gather: (Wr + i Wi) @ (hr + i hi)
        msg_r = torch.einsum("bnmd,bmd->bnd", w_r, h_real) - torch.einsum(
            "bnmd,bmd->bnd", w_i, h_imag
        )
        msg_i = torch.einsum("bnmd,bmd->bnd", w_r, h_imag) + torch.einsum(
            "bnmd,bmd->bnd", w_i, h_real
        )

        h_real = self.norm_real(h_real + self.proj_real(msg_r))
        h_imag = self.norm_imag(h_imag + self.proj_imag(msg_i))
        return h_real, h_imag


class GeometricHuygensMolecule(nn.Module):
    """Coordinate-driven Huygens stack → scalar molecular property."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_layers: int = 3,
        max_z: int = 20,
        cutoff_angstrom: float = 5.0,
        gamma: float = 0.5,
        omega: float = 0.3,
        dropout: float = 0.1,
        learnable_kernel_params: bool = True,
        n_rbf: int = 16,
        use_feature_distance: bool = False,
    ):
        super().__init__()
        self.cutoff = float(cutoff_angstrom)
        self.embed_dim = int(embed_dim)
        self.use_feature_distance = bool(use_feature_distance)
        self.source = AtomSecondarySources(embed_dim, max_z=max_z)
        self.medium = AtomicMediumDensity(embed_dim)
        self.layers = nn.ModuleList([
            GeometricHuygensConv(
                embed_dim,
                n_rbf=n_rbf,
                cutoff=cutoff_angstrom,
                gamma=gamma * (0.95 ** i),
                omega=omega * (1.05 ** i),
                learnable_kernel_params=learnable_kernel_params,
            )
            for i in range(int(num_layers))
        ])
        self.dropout = nn.Dropout(dropout)
        # Match M1 pooling + keep energy as second channel (still competitive head)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        scramble_geometry: bool = False,
        occlude_shell: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        if atom_mask is None:
            atom_mask = atomic_numbers > 0

        pos = positions
        if scramble_geometry:
            noise = torch.randn_like(pos) * self.cutoff
            pos = pos + noise

        h = self.source(atomic_numbers)
        h_real, h_imag = h, torch.zeros_like(h)

        for layer in self.layers:
            rho = self.medium(h_real)
            h_real, h_imag = layer(
                h_real,
                h_imag,
                pos,
                atom_mask,
                rho=rho,
                occlude_shell=occlude_shell,
                use_feature_distance=self.use_feature_distance,
                feat_for_distance=h_real if self.use_feature_distance else None,
            )
            h_real = self.dropout(h_real)
            h_imag = self.dropout(h_imag)

        energy = h_real.pow(2) + h_imag.pow(2)
        m = atom_mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp_min(1.0)
        mean_h = (h_real * m).sum(dim=1) / denom
        mean_e = (energy * m).sum(dim=1) / denom
        feat = torch.cat([mean_h, mean_e], dim=-1)
        y = self.head(feat).squeeze(-1)
        return {
            "y": y,
            "rho_mean": (rho.squeeze(-1) * atom_mask.float()).sum(dim=1) / denom.squeeze(-1),
            "energy": energy,
        }


class NoGeomAtomMLP(nn.Module):
    """M0 baseline: bag-of-atoms MLP (no geometry)."""

    def __init__(self, embed_dim: int = 64, max_z: int = 20, dropout: float = 0.1):
        super().__init__()
        self.source = AtomSecondarySources(embed_dim, max_z=max_z)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, positions, atomic_numbers, atom_mask=None, **_):
        if atom_mask is None:
            atom_mask = atomic_numbers > 0
        h = self.source(atomic_numbers)
        m = atom_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        y = self.head(pooled).squeeze(-1)
        return {"y": y}
