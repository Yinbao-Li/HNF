# -*- coding: utf-8 -*-

import torch

from hnf.inversion_1d import default_synth_model
from hnf.ray_paths import direct_ray_path


def test_direct_ray_path_endpoints():
    model = default_synth_model("cpu")
    x, z = direct_ray_path(model, "P", 10.0, 30.0)
    assert abs(float(x[0])) < 1e-6
    assert abs(float(z[0]) - 10.0) < 1e-6
    assert abs(float(x[-1]) - 30.0) < 1e-5
    assert abs(float(z[-1])) < 1e-6
    assert torch.all(z[1:] <= z[:-1])
