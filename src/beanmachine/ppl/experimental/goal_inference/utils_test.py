# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from beanmachine.facebook.goal_inference.utils import manhattan_distance


def test_manhattan_distance():
    assert manhattan_distance((1.0, 1.0), (1.0, 1.0)) == 0.0
    assert manhattan_distance((1.0, 2.0), (1.0, 1.0)) == 1.0
    assert manhattan_distance((1.0, 1.0), (2.0, 1.0)) == 1.0
    assert manhattan_distance((1.0, 1.0), (2.0, 2.0)) == 2.0
