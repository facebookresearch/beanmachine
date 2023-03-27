# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_definitions import (
    Direction,
    Gem,
    Item,
    Key,
)

from beanmachine.facebook.goal_inference.doors_keys_gems.parser import parse

####### Test Definition


@pytest.fixture
def key():
    k1 = Key("key1")
    return k1


@pytest.fixture
def gem():
    g1 = Gem("gem1")
    return g1


@pytest.fixture
def state():
    state = parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-1.pddl"
    )
    return state


@pytest.fixture
def right_direction():
    right_direction = Direction("right")
    return right_direction


def test_gem_name(gem):
    assert gem.name == "gem1"


def test_gem_class(gem):
    assert isinstance(gem, Gem) and isinstance(gem, Item)


def test_key_name(key):
    assert key.name == "key1"


def test_key_class(key):
    assert isinstance(key, Key) and isinstance(key, Item)


def test_direction_name(right_direction):
    assert right_direction.name == "right"


def test_direction_class(right_direction):
    assert isinstance(right_direction, Direction) and not isinstance(
        right_direction, Item
    )


def test_str(state):
    assert isinstance(str(state), str)
