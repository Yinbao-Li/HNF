# -*- coding: utf-8 -*-
import torch
from hnf.fluid_sota3d import FlowMRINetUnrolled3D, RecFNO3D
from hnf.fluid_spatial3d import Spatial3DFluidHNFReconstructor, curl_3d, unfold3d
from hnf.fluid_spatial4d import Spatial4DFluidHNFReconstructor
from hnf.fluid_synth3d import make_sample3d
from hnf.fluid_synth4d import make_sample4d


def test_sample3d_shapes():
    s = make_sample3d(d=8, h=8, w=8, seed=1)
    assert s["dense"].shape == (3, 8, 8, 8)
    assert s["mask"].shape == (1, 8, 8, 8)


def test_spatial3d_forward():
    m = Spatial3DFluidHNFReconstructor(d=8, h=8, w=8, embed_dim=16, kernel_size=3, predict_eta=False)
    x = torch.randn(1, 4, 8, 8, 8)
    y, _ = m(x, return_aux=True)
    assert y.shape == (1, 3, 8, 8, 8)


def test_spatial4d_forward():
    m = Spatial4DFluidHNFReconstructor(t_steps=4, d=6, h=8, w=8, embed_dim=16, kernel_size=3)
    x = torch.randn(1, 4, 4, 6, 8, 8)
    y, _ = m(x, return_aux=True)
    assert y.shape == (1, 3, 4, 6, 8, 8)


def test_literature_sota3d_forward():
    x = torch.randn(1, 4, 8, 8, 8)
    for cls in (RecFNO3D, FlowMRINetUnrolled3D):
        y = cls()(x)
        assert y.shape == (1, 3, 8, 8, 8)


def test_unfold3d():
    x = torch.randn(1, 2, 4, 4, 4)
    p = unfold3d(x, 3, 1)
    assert p.shape == (1, 2, 27, 4, 4, 4)
