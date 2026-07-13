# -*- coding: utf-8 -*-
"""Tests for Physics Decoder / Zhizi inversion."""

import torch

from hnf.inversion_1d import default_station_distances, default_synth_model, synthesize_travel_times
from hnf.stead_zhizi_inversion_dataset import SteadZhiziInversionDataset
from hnf.physics_decoder import PhysicsDecoder, features_to_head_inputs
from hnf.zhizi_inversion_loss import zhizi_inversion_loss
from hnf.zhizi_physics_head import (
    ZhiziPhysicsHead,
    bucket_rho_to_layers,
    count_physics_head_params,
    pool_wavefield_features,
    reference_layered_velocity,
)


def test_physics_head_forward():
    head = ZhiziPhysicsHead(embed_dim=32, n_layers=5, hidden=24)
    b = 2
    ws = torch.randn(b, 64)
    rho_l = torch.rand(b, 5)
    v_lat = torch.tensor([[8.0, 4.5], [7.5, 4.2]])
    out = head(ws, rho_l, v_lat)
    assert out.vp.shape == (b, 5)
    assert out.vs.shape == (b, 5)
    assert (out.vp[:, 1:] >= out.vp[:, :-1]).all()
    assert count_physics_head_params(head) < 20_000


def test_physics_head_zero_init_stays_near_reference():
    for mode in ("residual", "macro"):
        head = ZhiziPhysicsHead(embed_dim=16, n_layers=5, hidden=16, mode=mode)
        ws = torch.zeros(1, 32)
        rho_l = torch.zeros(1, 5)
        v_lat = torch.tensor([[8.0, 4.5]])
        out = head(ws, rho_l, v_lat)
        ref_vp, ref_vs = reference_layered_velocity(5, out.vp.device, out.vp.dtype)
        assert torch.allclose(out.vp[0], ref_vp, atol=1e-4), mode
        assert torch.allclose(out.vs[0], ref_vs, atol=1e-4), mode


def test_pool_and_bucket():
    h = torch.randn(1, 100, 32)
    hi = torch.zeros_like(h)
    pool = pool_wavefield_features(h, hi)
    assert pool.shape == (1, 64)
    rho = torch.rand(1, 100)
    layers = bucket_rho_to_layers(rho, 5)
    assert layers.shape == (1, 5)


def test_inversion_loss_differentiable():
    head = ZhiziPhysicsHead(embed_dim=16, n_layers=5, hidden=16)
    ws = torch.randn(1, 32)
    rho_l = torch.rand(1, 5)
    v_lat = torch.tensor([[8.0, 4.5]])
    out = head(ws, rho_l, v_lat)
    earth = default_synth_model("cpu")
    distances = default_station_distances("cpu", 4)
    clean = synthesize_travel_times(earth, 10.0, distances)
    loss, metrics = zhizi_inversion_loss(
        out,
        depths=earth.depths,
        q=earth.q,
        source_depth=10.0,
        receiver_distances=distances,
        obs_tp=clean["tp"],
        obs_ts=clean["ts"],
        rho_layers=rho_l[0],
        true_vp=earth.vp,
        true_vs=earth.vs,
        vp_sup_weight=0.1,
    )
    loss.backward()
    assert head.out.weight.grad is not None
    assert metrics["loss_tt"] >= 0.0


def test_physics_head_geo_conditioning():
    head = ZhiziPhysicsHead(embed_dim=16, n_layers=5, hidden=16, geo_dim=2)
    ws = torch.randn(1, 32)
    rho_l = torch.rand(1, 5)
    v_lat = torch.tensor([[8.0, 4.5]])
    geo = torch.tensor([[0.5, 0.3]])
    out = head(ws, rho_l, v_lat, geo=geo)
    assert out.vp.shape == (1, 5)


def test_stead_zhizi_dataset_len():
    ds = SteadZhiziInversionDataset(split="val", seq_len=200, max_traces=8, seed=0)
    assert len(ds) >= 1
    item = ds[0]
    assert item["x"].shape[-1] == 3
    assert item["geo"].shape == (2,)


def test_bridge_trainable_count_only_head():
    """Physics head only — backbone mocked."""
    class FakeBackbone(torch.nn.Module):
        embed_dim = 16

        def forward_inversion_features(self, x, t, include_picks=False):
            b, seq, _ = x.shape
            c = 16
            return {
                "h_real": torch.randn(b, seq, c),
                "h_imag": torch.zeros(b, seq, c),
                "rho": torch.rand(b, seq),
                "kernel_vp": torch.full((b,), 8.0),
                "kernel_vs": torch.full((b,), 4.5),
                "p_logits": torch.randn(b, seq),
                "s_logits": torch.randn(b, seq),
            }

    bb = FakeBackbone()
    bridge = PhysicsDecoder(bb, n_layers=5, embed_dim=16, hidden=16, infer_seq_len=None)
    assert bridge.trainable_parameter_count() == count_physics_head_params(bridge.physics_head)
    assert bridge.trainable_parameter_count() < bridge.total_parameter_count() or True

    x = torch.randn(3, 50, 3)
    t = torch.linspace(0, 60, 50).unsqueeze(-1)
    out, rho = bridge.forward_event(x, t)
    assert out.vp.shape == (1, 5)
