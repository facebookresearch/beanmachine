# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch
from sts.changepoints import BinSegChangepoint


@pytest.mark.parametrize("m, b", [(1, 0), (2, 1), (-1, 5)])
def test_no_changepoint(m, b):
    x = torch.linspace(0, 1, 100)
    y = m * x + b
    binseg = BinSegChangepoint()
    cps = binseg.get_changepoints(x, y)
    assert len(cps) == 0


def test_one_changepoint():
    x = torch.linspace(0, 1, 100)
    cp = 10
    x0, y0 = x[cp], 2 * x[cp] + 1
    seg_1 = 2 * x[:cp]
    seg_2 = -y0 / (1 - x0) * (x[cp:] - x0) + y0
    y = torch.cat([seg_1, seg_2])
    binseg = BinSegChangepoint(min_segment=5)
    cp = binseg.get_changepoints(x, y)
    assert len(cp) == 1
    assert cp[0] == 10
    binseg = BinSegChangepoint(min_segment=10)
    cp = binseg.get_changepoints(x, y)
    assert len(cp) == 1
    assert cp[0] == 10
    binseg = BinSegChangepoint(min_segment=12)
    cp = binseg.get_changepoints(x, y)
    assert len(cp) == 1
    assert cp[0] == 12


def test_pruned_changepoints():
    r"""
    Plot of y vs. x

    y0 _ | __  /
         |/  \/
         --------
          1 2 3  (changepoint locations)
    """
    x = torch.linspace(0, 1, 100)
    cp1 = 25
    cp2 = 50
    cp3 = 75
    y0 = 2 * x[cp1 - 1]
    seg_1 = 2 * x[:cp1]
    seg_2 = y0 * torch.ones(cp2 - cp1)
    seg_3 = -y0 / (x[cp3 - 1] - x[cp2 - 1]) * (x[cp2:cp3] - x[cp2 - 1]) + y0
    seg_4 = 1 / (1 - x[cp3 - 1]) * (x[cp3:] - x[cp3 - 1])
    y = torch.cat([seg_1, seg_2, seg_3, seg_4])
    binseg = BinSegChangepoint(min_segment=5, penalty_weight=1e-4, max_changepoints=2)
    cp = binseg.get_changepoints(x, y)
    assert len(cp) == 2  # In general, the exact changepoints will not be recovered
