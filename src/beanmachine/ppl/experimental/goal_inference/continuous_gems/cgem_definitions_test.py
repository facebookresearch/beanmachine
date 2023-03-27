# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses

from beanmachine.facebook.goal_inference.continuous_gems.cgem_definitions import (
    CGemState,
    Gem,
)


def test_initialize_gem():
    gem1 = Gem("gem1")
    gem2 = Gem("gem2")
    assert gem1.name == "gem1"
    assert gem1.name != gem2.name


def test_initialize_cgem_state():
    state_one = CGemState(
        ("has", "gem1"),
        5.0,
        5.0,
        {},
        0.5,
        {},
        {},
        set(),
        0.5,
        0.5,
        2.0,
        2.0,
        50.0,
    )
    assert state_one.x == 2.0
    assert state_one.y == 2.0


def test_compare_cgem_state():
    state_one = CGemState(
        ("has", "gem1"),
        5.0,
        5.0,
        {},
        0.5,
        {},
        {},
        set(),
        0.5,
        0.5,
        2.0,
        2.0,
        50.0,
    )
    state_two = dataclasses.replace(state_one, x=state_one.x)
    state_three = dataclasses.replace(state_one, x=state_one.x + 1)
    assert state_one == state_two
    assert state_one != state_three
