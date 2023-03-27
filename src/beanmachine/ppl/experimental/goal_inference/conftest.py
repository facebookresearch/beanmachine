# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_domain import DKGDomain

from beanmachine.facebook.goal_inference.doors_keys_gems.parser import parse

from beanmachine.facebook.goal_inference.planner.astar import AstarPlanner

from beanmachine.facebook.goal_inference.planner.planner import get_plan_from_actions

from beanmachine.facebook.goal_inference.planner.stoch_astar import (
    StochasticAstarPlanner,
)

from beanmachine.facebook.goal_inference.utils import manhattan_gem_heuristic


@pytest.fixture(scope="module")
def dkg_domain():
    return DKGDomain()


@pytest.fixture(scope="module")
def dkg_state_one():
    return parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-1.pddl"
    )


@pytest.fixture(scope="module")
def dkg_state_two():
    return parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-2.pddl"
    )


@pytest.fixture(scope="module")
def dkg_state_three():
    return parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-3.pddl"
    )


@pytest.fixture(scope="module")
def dkg_state_four():
    return parse(
        "beanmachine/facebook/goal_inference/doors_keys_gems/test_problems/problem-4.pddl"
    )


@pytest.fixture(scope="function")
def dkg_astar_planner(dkg_domain):
    return AstarPlanner(dkg_domain, manhattan_gem_heuristic)


@pytest.fixture(scope="function")
def dkg_stoch_astar_planner(dkg_domain):
    return StochasticAstarPlanner(dkg_domain, manhattan_gem_heuristic, 0.1)


@pytest.fixture(scope="function")
def dkg_observation_set(dkg_domain, dkg_state_three):
    actions = [
        ["right"],
        ["right"],
        ["right"],
        ["up"],
        ["up"],
        ["left"],
        ["left"],
        ["left"],
        ["up"],
        ["up"],
        ["up"],
        ["up"],
        ["pickup", "key1"],
        ["right"],
        ["right"],
        ["unlock", "key1", "right"],
        ["right"],
        ["right"],
        ["up"],
    ]
    observed_plan = get_plan_from_actions(dkg_domain, dkg_state_three, actions)
    return observed_plan.states
