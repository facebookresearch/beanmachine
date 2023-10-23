# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import pytest
from beanmachine.facebook.goal_inference.continuous_gems.cgem_domain import CGemDomain
from beanmachine.facebook.goal_inference.continuous_gems.cgem_parse import parse


@pytest.fixture(scope="function")
def cgem_domain():
    return CGemDomain()


@pytest.fixture(scope="function")
def state_one():
    return parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-1.json"
    )


def test_domain_predicates(cgem_domain, state_one):
    assert not cgem_domain.predicates["has"](state_one, "gem1")
    assert cgem_domain.predicates["at"](state_one, "gem1", 4.0, 0.5)
    assert cgem_domain.predicates["at"](state_one, "gem1", 4.1, 0.6)
    assert not cgem_domain.predicates["at"](state_one, "gem1", 4.6, 0.6)
    assert cgem_domain.predicates["obstacle"](state_one, 2.0, 2.0)
    assert cgem_domain.predicates["obstacle"](state_one, 2.1, 2.1)
    assert not cgem_domain.predicates["obstacle"](state_one, 0.01, 0.01)


def test_domain_rotate(cgem_domain, state_one):
    assert cgem_domain.is_action_possible(state_one, "rotate", 10.0)
    assert cgem_domain.is_action_possible(state_one, "rotate", -10.0)
    new_state = cgem_domain.execute(state_one, "rotate", -10.0)
    assert math.isclose(new_state.angle, 350.0, abs_tol=0.1)
    new_state = cgem_domain.execute(new_state, "rotate", 15.0)
    assert math.isclose(new_state.angle, 5.0, abs_tol=0.1)
    new_state = cgem_domain.execute(new_state, "rotate", 15.0)
    assert math.isclose(new_state.angle, 20.0, abs_tol=0.1)


def test_domain_move(cgem_domain, state_one):
    new_state = cgem_domain.execute(state_one, "rotate", 180.0)
    assert not cgem_domain.is_action_possible(new_state, "move", 0.8)
    assert not cgem_domain.is_action_possible(new_state, "move", 10.0)
    assert cgem_domain.is_action_possible(new_state, "move", 0.1)
    new_state = cgem_domain.execute(new_state, "move", 0.1)
    assert math.isclose(new_state.x, state_one.x - 0.1, abs_tol=0.01)
    assert math.isclose(new_state.y, state_one.y, abs_tol=0.01)
    new_state_two = cgem_domain.execute(new_state, "rotate", -135.0)
    new_state_two = cgem_domain.execute(new_state_two, "move", 0.1)
    assert math.isclose(
        new_state_two.x, new_state.x + 0.1 * math.sqrt(2.0) / 2.0, abs_tol=0.01
    )
    assert math.isclose(
        new_state_two.y, new_state.y + 0.1 * math.sqrt(2.0) / 2.0, abs_tol=0.01
    )


def test_domain_pickup(cgem_domain, state_one):
    assert not cgem_domain.evaluate_goal(state_one)
    new_state = cgem_domain.execute(state_one, "rotate", 270.0)
    new_state = cgem_domain.execute(new_state, "move", 0.5)
    new_state = cgem_domain.execute(new_state, "rotate", 90.0)
    new_state = cgem_domain.execute(new_state, "move", 2.9)
    assert not cgem_domain.is_action_possible(new_state, "pickup", "gem2")
    new_state = cgem_domain.execute(new_state, "pickup", "gem1")
    assert cgem_domain.evaluate_goal(new_state)


def test_domain_next_possible_actions(cgem_domain, state_one):
    possible_actions = cgem_domain.get_possible_actions(state_one)
    assert len(possible_actions) == 2
    assert possible_actions[0][0] == "rotate"
    assert possible_actions[1][0] == "move"
    new_state = cgem_domain.execute(state_one, "move", 2.9)
    possible_actions = cgem_domain.get_possible_actions(new_state)
    assert possible_actions[0][0] == "rotate"
    assert possible_actions[-1] == ["pickup", "gem1"]
    new_state_two = cgem_domain.execute(state_one, "rotate", 270.0)
    new_state_two = cgem_domain.execute(new_state_two, "move", 0.74999)
    possible_actions = cgem_domain.get_possible_actions(new_state_two)
    assert len(possible_actions) == 1
    assert possible_actions[0][0] == "rotate"
