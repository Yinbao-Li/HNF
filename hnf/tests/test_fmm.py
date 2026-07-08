# -*- coding: utf-8 -*-
"""Tests for user's FMM implementation."""

import torch

from hnf.fmm import DirectPropagation, FastMultipoleMethod
from hnf.kernel import HuygensKernel


def test_fmm_tree_builds():
    kernel = HuygensKernel(causal=False)
    fmm = FastMultipoleMethod(kernel, max_leaf_size=8)
    x = torch.rand(64, 2)
    tree = fmm.build_tree(x)
    assert len(tree["nodes"]) >= 1
    assert tree["root"] == 0


def test_direct_propagation_shape():
    kernel = HuygensKernel(causal=False)
    direct = DirectPropagation(kernel)
    x = torch.rand(20, 2)
    sources = torch.randn(20, 1)
    out = direct.forward(x, sources)
    assert out.shape == (20, 1)


def test_fmm_estimate_complexity():
    kernel = HuygensKernel(causal=False)
    fmm = FastMultipoleMethod(kernel, max_leaf_size=8, expansion_order=4)
    assert fmm.estimate_complexity(100) > 0
