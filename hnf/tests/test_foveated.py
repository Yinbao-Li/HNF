# -*- coding: utf-8 -*-
"""Smoke tests for the foveated active-perception stack."""

from __future__ import annotations

import torch

from hnf.foveated import (
    CausalMemory,
    FoveatedEngine,
    FoveaProcessor,
    PeripheralScanner,
    Scheduler,
)
from hnf.foveated.engine import visualize_trajectory_ascii
from hnf.foveated.training import FoveatedTrainConfig, stage2_joint_loss


def _synthetic_wave(batch: int = 2, seq_len: int = 6000) -> torch.Tensor:
    """(B,3,T) with P/S-like bursts."""
    x = 0.05 * torch.randn(batch, 3, seq_len)
    for b in range(batch):
        p = 1500 + b * 200
        s = p + 800
        t = torch.arange(seq_len)
        x[b, :, :] += 1.2 * torch.exp(-0.5 * ((t - p) / 15.0) ** 2)
        x[b, :, :] += 0.9 * torch.exp(-0.5 * ((t - s) / 25.0) ** 2)
    return x


def test_peripheral_scanner_shapes():
    scanner = PeripheralScanner(seq_len=6000, detector="energy")
    x = _synthetic_wave()
    heat, cands = scanner(x)
    assert heat.shape == (2, 6000)
    assert heat.min() >= 0 and heat.max() <= 1
    assert isinstance(cands, list) and len(cands) == 2
    assert all(hasattr(r, "start") for regs in cands for r in regs)


def test_peripheral_sparse_huygens():
    scanner = PeripheralScanner(seq_len=6000, detector="sparse_huygens", stride=10)
    x = _synthetic_wave(batch=1)
    heat, cands = scanner(x)
    assert heat.shape == (1, 6000)
    assert cands is not None


def test_fovea_processor_fallback():
    fov = FoveaProcessor(seq_len=6000, default_window_size=800)
    x = _synthetic_wave(batch=1)
    out = fov(x, focus_index=1800, window_size=800)
    assert out.p_logits.shape == (1, 800)
    assert out.s_logits.shape == (1, 800)
    assert 0 <= out.window_start < out.window_end <= 6000
    assert out.p_idx_global.shape == (1,)


def test_scheduler_and_memory():
    mem = CausalMemory()
    sched = Scheduler(seq_len=6000)
    fov = FoveaProcessor(seq_len=6000)
    x = _synthetic_wave(batch=1)
    heat, cands = PeripheralScanner(seq_len=6000)(x)
    d0 = sched(heat[0], mem.graph, candidates=cands[0])
    assert 0 <= d0.focus_index < 6000
    assert d0.window_size in {200, 400, 800, 1200, 1500}
    out = fov(x, d0.focus_index, d0.window_size)
    node = mem.remember(out)
    assert node.node_id == 0
    d1 = sched(heat[0], mem.graph, candidates=cands[0])
    assert isinstance(d1.reason, str)


def test_foveated_engine_forward():
    engine = FoveatedEngine(seq_len=6000, max_gazes=4)
    x = _synthetic_wave(batch=2)
    out = engine(x, max_gazes=4)
    assert out.heatmap.shape == (2, 6000)
    assert out.p_logits.shape == (2, 6000)
    assert out.s_logits.shape == (2, 6000)
    assert out.p_idx.shape == (2,)
    assert out.n_gazes.shape == (2,)
    assert int(out.n_gazes.max()) <= 4
    assert len(out.trajectory) == 2
    ascii_viz = visualize_trajectory_ascii(out.trajectory[0])
    assert isinstance(ascii_viz, str) and len(ascii_viz) == 80


def test_stage2_loss_smoke():
    engine = FoveatedEngine(seq_len=6000, max_gazes=3)
    x = _synthetic_wave(batch=1)
    out = engine(x, max_gazes=3)
    p_tgt = torch.zeros_like(out.p_logits)
    s_tgt = torch.zeros_like(out.s_logits)
    p_tgt[0, 1500] = 1.0
    s_tgt[0, 2300] = 1.0
    loss, stats = stage2_joint_loss(engine, out, p_tgt, s_tgt, FoveatedTrainConfig(max_gazes=3))
    assert torch.isfinite(loss)
    assert "loss_pick" in stats


if __name__ == "__main__":
    test_peripheral_scanner_shapes()
    test_peripheral_sparse_huygens()
    test_fovea_processor_fallback()
    test_scheduler_and_memory()
    test_foveated_engine_forward()
    test_stage2_loss_smoke()
    print("all foveated smoke tests passed")
