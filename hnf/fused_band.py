# -*- coding: utf-8 -*-
"""Fused causal band apply for Huygens kernels (Triton).

Computes, without materialising the (B, W, N, D) neighbour stack,

    out[b, i, :] = sum_{l=1..W, i-l>=0} a[l] * s(b, i, l) * h[b, i-l, :]
    s(b, i, l)   = exp(-(rho[b, i] + rho[b, i-l]) / 2 * lag[l])   (1 if rho is None)

with complex ``a`` (W,) and ``h`` (B, N, D). Gradients flow to ``h``, ``a`` and ``rho``.
Set ``HNF_FUSED_BAND=0`` to force the reference PyTorch path.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

_ENABLED = os.environ.get("HNF_FUSED_BAND", "1") != "0"
_MAX_D = 256
_BI_COEF = int(os.environ.get("HNF_FUSED_BI_COEF", "16"))
_LC_COEF = int(os.environ.get("HNF_FUSED_LC_COEF", "32"))
_BI_FWD = int(os.environ.get("HNF_FUSED_BI", "16"))
_BD_FWD = int(os.environ.get("HNF_FUSED_BD", "32"))


def set_enabled(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def can_use(h: torch.Tensor) -> bool:
    return (
        _ENABLED
        and _HAS_TRITON
        and h.is_cuda
        and h.is_complex()
        and h.dim() == 3
        and h.size(-1) <= _MAX_D
    )


if _HAS_TRITON:

    @triton.jit
    def _band_fwd(
        hr_ptr, hi_ptr, ar_ptr, ai_ptr, lag_ptr, rho_ptr, or_ptr, oi_ptr,
        N, D, W,
        HAS_RHO: tl.constexpr, BI: tl.constexpr, BD: tl.constexpr,
    ):
        pid_i = tl.program_id(0)
        pid_d = tl.program_id(1)
        b = tl.program_id(2)
        offs_i = pid_i * BI + tl.arange(0, BI)
        offs_d = pid_d * BD + tl.arange(0, BD)
        mi = offs_i < N
        md = offs_d < D
        base = b * N * D
        if HAS_RHO:
            rho_i = tl.load(rho_ptr + b * N + offs_i, mask=mi, other=0.0)
        acc_r = tl.zeros((BI, BD), dtype=tl.float32)
        acc_i = tl.zeros((BI, BD), dtype=tl.float32)
        for l in range(1, W + 1):
            ar = tl.load(ar_ptr + l - 1)
            ai = tl.load(ai_ptr + l - 1)
            j = offs_i - l
            mj = mi & (j >= 0)
            m2 = mj[:, None] & md[None, :]
            ptr = base + j[:, None] * D + offs_d[None, :]
            hr = tl.load(hr_ptr + ptr, mask=m2, other=0.0)
            hi = tl.load(hi_ptr + ptr, mask=m2, other=0.0)
            if HAS_RHO:
                lag = tl.load(lag_ptr + l - 1)
                rho_j = tl.load(rho_ptr + b * N + j, mask=mj, other=0.0)
                s = tl.exp(-(rho_i + rho_j) * 0.5 * lag)
                cr = (ar * s)[:, None]
                ci = (ai * s)[:, None]
            else:
                cr = ar
                ci = ai
            acc_r += cr * hr - ci * hi
            acc_i += cr * hi + ci * hr
        optr = base + offs_i[:, None] * D + offs_d[None, :]
        om = mi[:, None] & md[None, :]
        tl.store(or_ptr + optr, acc_r, mask=om)
        tl.store(oi_ptr + optr, acc_i, mask=om)

    @triton.jit
    def _band_bwd_h(
        gor_ptr, goi_ptr, ar_ptr, ai_ptr, lag_ptr, rho_ptr, ghr_ptr, ghi_ptr,
        N, D, W,
        HAS_RHO: tl.constexpr, BI: tl.constexpr, BD: tl.constexpr,
    ):
        # Iterate over *source* rows j; receivers are i = j + l.
        pid_j = tl.program_id(0)
        pid_d = tl.program_id(1)
        b = tl.program_id(2)
        offs_j = pid_j * BI + tl.arange(0, BI)
        offs_d = pid_d * BD + tl.arange(0, BD)
        mj = offs_j < N
        md = offs_d < D
        base = b * N * D
        if HAS_RHO:
            rho_j = tl.load(rho_ptr + b * N + offs_j, mask=mj, other=0.0)
        acc_r = tl.zeros((BI, BD), dtype=tl.float32)
        acc_i = tl.zeros((BI, BD), dtype=tl.float32)
        for l in range(1, W + 1):
            ar = tl.load(ar_ptr + l - 1)
            ai = tl.load(ai_ptr + l - 1)
            i = offs_j + l
            mi = mj & (i < N)
            m2 = mi[:, None] & md[None, :]
            ptr = base + i[:, None] * D + offs_d[None, :]
            gr = tl.load(gor_ptr + ptr, mask=m2, other=0.0)
            gi = tl.load(goi_ptr + ptr, mask=m2, other=0.0)
            if HAS_RHO:
                lag = tl.load(lag_ptr + l - 1)
                rho_i = tl.load(rho_ptr + b * N + i, mask=mi, other=0.0)
                s = tl.exp(-(rho_i + rho_j) * 0.5 * lag)
                cr = (ar * s)[:, None]
                ci = (ai * s)[:, None]
            else:
                cr = ar
                ci = ai
            acc_r += cr * gr + ci * gi
            acc_i += cr * gi - ci * gr
        optr = base + offs_j[:, None] * D + offs_d[None, :]
        om = mj[:, None] & md[None, :]
        tl.store(ghr_ptr + optr, acc_r, mask=om)
        tl.store(ghi_ptr + optr, acc_i, mask=om)

    @triton.jit
    def _band_bwd_coef(
        hr_ptr, hi_ptr, gor_ptr, goi_ptr, ar_ptr, ai_ptr, lag_ptr, rho_ptr,
        par_ptr, pai_ptr, grho_ptr,
        N, D, W,
        HAS_RHO: tl.constexpr, BI: tl.constexpr, BD: tl.constexpr, LC: tl.constexpr,
    ):
        # One program covers BI receivers and the full feature dim (BD >= D).
        # Per-program coefficient partial sums go to par/pai[(pid), l] and are reduced on the host;
        # scalar atomics onto W shared addresses serialised badly.
        pid_i = tl.program_id(0)
        b = tl.program_id(1)
        chunk = tl.program_id(2)
        prog = b * tl.num_programs(0) + pid_i
        l_lo = chunk * LC + 1
        l_hi = tl.minimum(l_lo + LC, W + 1)
        offs_i = pid_i * BI + tl.arange(0, BI)
        offs_d = tl.arange(0, BD)
        mi = offs_i < N
        md = offs_d < D
        base = b * N * D
        gptr = base + offs_i[:, None] * D + offs_d[None, :]
        gm = mi[:, None] & md[None, :]
        gr = tl.load(gor_ptr + gptr, mask=gm, other=0.0)
        gi = tl.load(goi_ptr + gptr, mask=gm, other=0.0)
        if HAS_RHO:
            rho_i = tl.load(rho_ptr + b * N + offs_i, mask=mi, other=0.0)
            grho_i = tl.zeros((BI,), dtype=tl.float32)
        for l in range(l_lo, l_hi):
            j = offs_i - l
            mj = mi & (j >= 0)
            m2 = mj[:, None] & md[None, :]
            ptr = base + j[:, None] * D + offs_d[None, :]
            hr = tl.load(hr_ptr + ptr, mask=m2, other=0.0)
            hi = tl.load(hi_ptr + ptr, mask=m2, other=0.0)
            p = tl.sum(hr * gr + hi * gi, axis=1)
            q = tl.sum(hr * gi - hi * gr, axis=1)
            if HAS_RHO:
                lag = tl.load(lag_ptr + l - 1)
                rho_j = tl.load(rho_ptr + b * N + j, mask=mj, other=0.0)
                s = tl.exp(-(rho_i + rho_j) * 0.5 * lag)
                s = tl.where(mj, s, 0.0)
                ar = tl.load(ar_ptr + l - 1)
                ai = tl.load(ai_ptr + l - 1)
                t = (ar * p + ai * q) * s * (-0.5 * lag)
                grho_i += t
                tl.atomic_add(grho_ptr + b * N + j, t, mask=mj)
            else:
                s = tl.where(mj, 1.0, 0.0)
            tl.store(par_ptr + prog * W + l - 1, tl.sum(s * p, axis=0))
            tl.store(pai_ptr + prog * W + l - 1, tl.sum(s * q, axis=0))
        if HAS_RHO:
            tl.atomic_add(grho_ptr + b * N + offs_i, grho_i, mask=mi)


def _launch_cfg(D: int):
    return _BI_FWD, min(_BD_FWD, triton.next_power_of_2(D))


class _FusedBand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hr, hi, ar, ai, lag, rho):
        B, N, D = hr.shape
        W = ar.numel()
        has_rho = rho is not None
        rho_t = rho if has_rho else hr.new_zeros(1)
        o_r = torch.empty_like(hr)
        o_i = torch.empty_like(hr)
        BI, BD = _launch_cfg(D)
        grid = (triton.cdiv(N, BI), triton.cdiv(D, BD), B)
        _band_fwd[grid](
            hr, hi, ar, ai, lag, rho_t, o_r, o_i, N, D, W,
            HAS_RHO=has_rho, BI=BI, BD=BD,
        )
        ctx.has_rho = has_rho
        ctx.save_for_backward(hr, hi, ar, ai, lag, rho_t)
        return o_r, o_i

    @staticmethod
    def backward(ctx, go_r, go_i):
        hr, hi, ar, ai, lag, rho_t = ctx.saved_tensors
        has_rho = ctx.has_rho
        B, N, D = hr.shape
        W = ar.numel()
        go_r = go_r.contiguous() if go_r is not None else torch.zeros_like(hr)
        go_i = go_i.contiguous() if go_i is not None else torch.zeros_like(hr)

        g_hr = torch.empty_like(hr)
        g_hi = torch.empty_like(hr)
        BI, BD = _launch_cfg(D)
        grid = (triton.cdiv(N, BI), triton.cdiv(D, BD), B)
        _band_bwd_h[grid](
            go_r, go_i, ar, ai, lag, rho_t, g_hr, g_hi, N, D, W,
            HAS_RHO=has_rho, BI=BI, BD=BD,
        )

        g_rho = torch.zeros_like(rho_t) if has_rho else rho_t
        BIc = _BI_COEF
        BDc = triton.next_power_of_2(D)
        n_prog = triton.cdiv(N, BIc) * B
        p_ar = torch.empty(n_prog, W, device=hr.device, dtype=torch.float32)
        p_ai = torch.empty_like(p_ar)
        _band_bwd_coef[(triton.cdiv(N, BIc), B, triton.cdiv(W, _LC_COEF))](
            hr, hi, go_r, go_i, ar, ai, lag, rho_t, p_ar, p_ai, g_rho, N, D, W,
            HAS_RHO=has_rho, BI=BIc, BD=BDc, LC=_LC_COEF,
        )
        return g_hr, g_hi, p_ar.sum(0), p_ai.sum(0), None, (g_rho if has_rho else None)


def band_apply(
    h: torch.Tensor,
    coef: torch.Tensor,
    lags: torch.Tensor,
    rho: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused causal band apply. h: complex (B,N,D); coef: (W,) real/complex; lags: (W,) sec;
    rho: (B,N) or (B,N,1) or None. Returns complex (B,N,D)."""
    hr = h.real.float().contiguous()
    hi = h.imag.float().contiguous()
    if coef.is_complex():
        ar = coef.real.float().contiguous()
        ai = coef.imag.float().contiguous()
    else:
        ar = coef.float().contiguous()
        ai = torch.zeros_like(ar)
    lag = lags.detach().float().contiguous()
    if rho is not None:
        if rho.dim() == 3:
            rho = rho.squeeze(-1)
        rho = rho.float().contiguous()
    o_r, o_i = _FusedBand.apply(hr, hi, ar, ai, lag, rho)
    return torch.complex(o_r, o_i)


def band_apply_reference(
    h: torch.Tensor,
    coef: torch.Tensor,
    lags: torch.Tensor,
    rho: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Straightforward PyTorch version of ``band_apply`` (for tests)."""
    B, N, D = h.shape
    W = coef.numel()
    out = torch.zeros_like(h)
    if rho is not None and rho.dim() == 3:
        rho = rho.squeeze(-1)
    for l in range(1, W + 1):
        if l >= N:
            break
        src = h[:, : N - l]
        c = coef[l - 1]
        if rho is not None:
            s = torch.exp(-(rho[:, l:] + rho[:, : N - l]) * 0.5 * lags[l - 1])
            term = (c * s).unsqueeze(-1) * src
        else:
            term = c * src
        out = out + torch.nn.functional.pad(term, (0, 0, l, 0))
    return out
