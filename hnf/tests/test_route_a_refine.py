# -*- coding: utf-8 -*-
from hnf.route_a_refine import RouteARow, build_verdict


def test_verdict_zhizi_better_init():
    rows = [
        RouteARow(0, 0.5, 1.0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 25),
        RouteARow(1, 0.4, 0.9, 0.18, 0.19, 0, 0, 0, 0, 0, 0, 25),
    ]
    v = build_verdict(rows)
    assert v.init_rmse_ratio < 0.85
    assert v.physically_meaningful
