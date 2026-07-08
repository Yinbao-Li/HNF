# -*- coding: utf-8 -*-
"""Tests for STEAD picking metrics."""

import torch

from hnf.picking_metrics import EvalAccumulator, finalize_metrics, update_picking_counts


def test_strict_picking_threshold_and_detection_gate():
    acc = EvalAccumulator()
    probs = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.1]], dtype=torch.float32)
    det_pred = torch.tensor([True, False])
    det_true = torch.tensor([True, True])
    valid = torch.tensor([True, True])
    gt_idx = torch.tensor([1, 2])

    update_picking_counts(
        acc.p,
        probs,
        det_pred,
        det_true,
        valid,
        gt_idx,
        pick_threshold=0.5,
        tol_bins=1,
        seq_len=60,
    )

    metrics = finalize_metrics(acc)
    assert acc.p.tp == 1
    assert acc.p.fn == 1
    assert abs(metrics["p_f1"] - 2 / 3) < 1e-6
    assert metrics["p_mae_sec"] == 0.0
