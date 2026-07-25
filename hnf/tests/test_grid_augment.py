# -*- coding: utf-8 -*-
"""Grid-resampling augmentation must move every time label with the waveform."""

from __future__ import annotations

import random

import pytest
import torch

from hnf.grid_augment import (
    parse_grid_lens,
    resample_batch_to_grid,
    rescale_index,
    sample_grid_len,
)


def _batch(b: int = 3, t: int = 800) -> dict[str, torch.Tensor]:
    p_idx = torch.tensor([100, 400, -1])[:b]
    s_idx = torch.tensor([200, 600, -1])[:b]
    sigma = max(1.0, 0.4 * t / 60.0)
    from hnf.stead_picking_dataset import gaussian_pick_label

    return {
        "x": torch.randn(b, t, 3),
        "t": torch.linspace(0, 60, t).view(1, t, 1).expand(b, t, 1).contiguous(),
        "det": torch.tensor([1.0, 1.0, 0.0])[:b],
        "p_idx": p_idx,
        "s_idx": s_idx,
        "p_valid": torch.tensor([1.0, 1.0, 0.0])[:b],
        "s_valid": torch.tensor([1.0, 1.0, 0.0])[:b],
        "p_target": torch.stack([gaussian_pick_label(int(i), t, sigma) for i in p_idx]),
        "s_target": torch.stack([gaussian_pick_label(int(i), t, sigma) for i in s_idx]),
    }


@pytest.mark.parametrize("out_len", [200, 400, 1200])
def test_shapes_and_time_axis(out_len: int) -> None:
    b = _batch()
    out = resample_batch_to_grid(b, out_len)
    assert out["x"].shape == (3, out_len, 3)
    assert out["t"].shape == (3, out_len, 1)
    assert torch.isfinite(out["x"]).all()
    assert out["t"][0, 0, 0].item() == pytest.approx(0.0)
    assert out["t"][0, -1, 0].item() == pytest.approx(60.0)
    assert out["p_target"].shape == (3, out_len)


@pytest.mark.parametrize("out_len", [200, 400, 1200, 6000])
def test_pick_time_in_seconds_is_preserved(out_len: int) -> None:
    b = _batch()
    out = resample_batch_to_grid(b, out_len)
    for key in ("p_idx", "s_idx"):
        for old, new in zip(b[key], out[key]):
            if int(old) < 0:
                continue
            old_sec = float(old) * 60.0 / 800.0
            new_sec = float(new) * 60.0 / out_len
            # within one bin of the coarser of the two grids
            assert abs(new_sec - old_sec) <= 60.0 / min(800, out_len)


def test_absent_pick_sentinel_survives() -> None:
    b = _batch()
    out = resample_batch_to_grid(b, 400)
    assert int(out["p_idx"][2]) == -1
    assert int(out["s_idx"][2]) == -1
    assert float(out["p_target"][2].max()) == 0.0


def test_target_peak_sits_on_the_remapped_index() -> None:
    b = _batch()
    out = resample_batch_to_grid(b, 400)
    for row in range(2):
        assert int(out["p_target"][row].argmax()) == int(out["p_idx"][row])
        assert float(out["p_target"][row].max()) == pytest.approx(1.0)


def test_identity_when_length_matches() -> None:
    b = _batch(t=400)
    assert resample_batch_to_grid(b, 400) is b


def test_rescale_index_clamps_within_grid() -> None:
    idx = torch.tensor([0, 799, -1])
    out = rescale_index(idx, 800, 200)
    assert int(out[0]) == 0
    assert int(out[1]) == 199
    assert int(out[2]) == -1


def test_parse_grid_lens() -> None:
    assert parse_grid_lens("400, 800,400") == [400, 800]
    assert parse_grid_lens("") == []
    with pytest.raises(ValueError):
        parse_grid_lens("8,400")


def test_sample_grid_len_respects_prob() -> None:
    rng = random.Random(0)
    assert sample_grid_len([400], prob=0.0, base_len=800, rng=rng) == 800
    assert sample_grid_len([], prob=1.0, base_len=800, rng=rng) == 800
    assert sample_grid_len([400], prob=1.0, base_len=800, rng=rng) == 400
    drawn = {
        sample_grid_len([400, 1200], prob=0.5, base_len=800, rng=rng)
        for _ in range(200)
    }
    assert drawn == {400, 800, 1200}
