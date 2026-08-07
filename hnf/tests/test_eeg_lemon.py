# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from hnf.eeg_lemon import age_bin_midpoint, lemon_sex_code


def test_age_bin_midpoint():
    assert abs(age_bin_midpoint("20-25") - 22.5) < 1e-9
    assert abs(age_bin_midpoint("65-70") - 67.5) < 1e-9
    assert abs(age_bin_midpoint("70–75") - 72.5) < 1e-9
    assert np.isnan(age_bin_midpoint(""))


def test_lemon_sex_code():
    assert lemon_sex_code("1") == "F"
    assert lemon_sex_code(2) == "M"
    assert lemon_sex_code("M") == "M"
    assert lemon_sex_code("") == ""
