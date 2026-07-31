# -*- coding: utf-8 -*-
"""Smoke tests for EEG-native geometry + forward pass."""

from __future__ import annotations

import torch

from hnf.eeg_geometry import electrode_xyz, pairwise_chord_distance, region_index_masks
from hnf.eeg_native_model import (
    EEGHNFNativeClassifier,
    RegionalEnergy,
    SpatialAnisoDiffusionMix,
    SpatialHuygensMix,
)


def test_electrode_geometry_shapes():
    xyz = electrode_xyz()
    assert xyz.shape == (19, 3)
    d = pairwise_chord_distance(xyz)
    assert d.shape == (19, 19)
    assert abs(float(d[0, 0])) < 1e-5
    assert float(d.max()) > 0.5
    masks = region_index_masks()
    assert masks["frontal"].sum() == 7
    assert masks["temporal"].sum() == 4


def test_spatial_mix_preserves_shape():
    mix = SpatialHuygensMix()
    x = torch.randn(2, 19, 128)
    y = mix(x)
    assert y.shape == x.shape
    reg = RegionalEnergy()
    assert reg(x).shape == (2, 6)


def test_spatial_aniso_diffusion_mix():
    mix = SpatialAnisoDiffusionMix(rhythm_phase=True)
    mix0 = SpatialAnisoDiffusionMix(rhythm_phase=False)
    x = torch.randn(2, 19, 64)
    y = mix(x)
    y0 = mix0(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, y0, atol=1e-5)
    p = mix.collect_params()
    assert p["tau0"] > 0
    assert p["D_eig0"] > 0


def test_native_forward_and_aux():
    model = EEGHNFNativeClassifier(seq_len=128, sample_rate=128, embed_dim=32)
    x = torch.randn(2, 19, 128)
    logits, aux = model(x, return_aux=True)
    assert logits.shape == (2, 3)
    assert aux["rho"].shape[0] == 2
    assert aux["band_proxy"].shape == (2, 4)
    assert aux["region_energy"].shape == (2, 6)
    assert "delta_env" in aux
    kp = model.collect_kernel_params()
    assert "spatial" in kp
    assert "theta_layer0" in kp
    assert "delta_layer0" in kp
    assert kp["theta_layer0"]["omega"] > 5.0
    assert kp["alpha_layer0"]["omega"] > 10.0
    assert kp["delta_layer0"]["omega"] > 2.0


def test_native_aniso_diffusion_phase_ablation_forward():
    x = torch.randn(2, 19, 128)
    m1 = EEGHNFNativeClassifier(
        seq_len=128,
        sample_rate=128,
        embed_dim=32,
        principle="aniso_diffusion",
        rhythm_phase=True,
    )
    m0 = EEGHNFNativeClassifier(
        seq_len=128,
        sample_rate=128,
        embed_dim=32,
        principle="aniso_diffusion",
        rhythm_phase=False,
    )
    y1 = m1(x)
    y0 = m0(x)
    assert y1.shape == (2, 3) and y0.shape == (2, 3)
    assert isinstance(m1.spatial, SpatialAnisoDiffusionMix)
    assert m1.theta.stack.layers[0].kernel.rhythm_phase is True
    assert m0.theta.stack.layers[0].kernel.rhythm_phase is False
    kp = m1.collect_kernel_params()
    assert "D_eig0" in kp["spatial"]
    assert kp["theta_layer0"]["principle"] == 1.0
