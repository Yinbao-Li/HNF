# -*- coding: utf-8 -*-
"""Smoke tests for DifferentiableKernelBank."""

from __future__ import annotations

import torch

from hnf.kernel_bank import DifferentiableKernelBank, KernelBankWaveBlock


def test_bank_forward_shapes():
    B, T, D, N = 2, 32, 8, 4
    bank = DifferentiableKernelBank(
        N,
        gamma=0.8,
        omega=0.5,
        wave_speed=6.0,
        local_window_sec=5.0,
        sparse_band=True,
        principle="huygens",
        top_m=2,
    )
    h_r = torch.randn(B, T, D)
    h_i = torch.zeros_like(h_r)
    t = torch.linspace(0, 60, T).view(1, T, 1).expand(B, T, 1)
    h_c = torch.complex(h_r, h_i)
    out = bank.forward_apply(h_c, h_r, t=t)
    assert out.shape == h_c.shape
    assert torch.isfinite(out.real).all()


def test_regularizers_and_schedule():
    bank = DifferentiableKernelBank(6, sparse_band=True, principle="huygens", top_m=3)
    h = torch.randn(2, 16, 4)
    st = bank.schedule_state(epoch=5, total_epochs=100)
    assert st.phase == "differentiate"
    regs = bank.bank_regularizers(h, st)
    assert "total" in regs
    assert torch.isfinite(regs["total"])
    st2 = bank.schedule_state(epoch=50, total_epochs=100)
    assert st2.phase == "merge"
    st3 = bank.schedule_state(epoch=80, total_epochs=100)
    assert st3.phase == "lock"
    bank.capture_role_anchors()
    assert float(bank.role_anchor_valid.sum()) >= 1.0


def test_merge_rollback():
    bank = DifferentiableKernelBank(4, sparse_band=True, principle="huygens", top_m=0)
    bank.apply_soft_merge(0, 1)
    assert float(bank.alive_probs()[1]) < 0.1
    ok = bank.rollback_last_merge()
    assert ok
    assert float(bank.alive_probs()[1]) > 0.4


def test_waveblock_forward():
    block = KernelBankWaveBlock(dim=8, n_kernels=3, sparse_band=True, principle="huygens", top_m=2)
    h_r = torch.randn(1, 24, 8)
    h_i = torch.zeros_like(h_r)
    t = torch.linspace(0, 60, 24).view(1, 24, 1)
    o_r, o_i = block(h_r, h_i, t=t)
    assert o_r.shape == h_r.shape
    assert torch.isfinite(o_r).all()
