# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses
import math

from typing import Dict, List

import pytest

import torch

from beanmachine.facebook.goal_inference.agent.boundedly_rational_agent import (
    BoundedRationalAgent,
)

from beanmachine.facebook.goal_inference.agent.observation_model import (
    apply_noise_to_states,
)
from beanmachine.facebook.goal_inference.continuous_gems.cgem_domain import CGemDomain
from beanmachine.facebook.goal_inference.continuous_gems.cgem_observation_model import (
    CGemObservation,
)

from beanmachine.facebook.goal_inference.continuous_gems.cgem_parse import parse

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_observation_model import (
    DKGObservation,
)

from beanmachine.facebook.goal_inference.inference.sequential_inverse_plan_search import (
    SIPS,
)
from beanmachine.facebook.goal_inference.planner.stoch_astar import (
    StochasticAstarPlanner,
)
from beanmachine.facebook.goal_inference.utils import manhattan_gem_heuristic


@pytest.fixture(scope="function")
def dkg_SIPS_no_noise(dkg_stoch_astar_planner):
    return SIPS(
        dkg_stoch_astar_planner,
        r=1000,
        p=0.5,
    )


@pytest.fixture(scope="function")
def dkg_SIPS_noise(dkg_stoch_astar_planner):
    return SIPS(
        dkg_stoch_astar_planner,
        DKGObservation(),
        r=1000,
        p=0.5,
    )


def get_last_inference_prediction(infer_data: List[Dict[str, float]]):
    """Returns the time step of posterior with nonzero weights"""
    time_step = len(infer_data) - 1
    while time_step > 0:
        for goal_id in infer_data[time_step]:
            if infer_data[time_step][goal_id] != 0:
                return infer_data[time_step]
        time_step -= 1
    return infer_data[0]


def test_dkg_problem_one_SIPS_no_noise(
    dkg_astar_planner, dkg_state_one, dkg_SIPS_no_noise
):
    (
        plan,
        solved,
    ) = dkg_astar_planner.generate_plan(dkg_state_one)
    observations = plan.states
    infered_goal = dkg_SIPS_no_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 1.0, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_two_SIPS_no_noise(
    dkg_astar_planner, dkg_state_two, dkg_SIPS_no_noise
):
    (
        plan,
        solved,
    ) = dkg_astar_planner.generate_plan(dkg_state_two)
    observations = plan.states
    infered_goal = dkg_SIPS_no_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.5, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.5, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_three_SIPS_no_noise(
    dkg_astar_planner, dkg_state_three, dkg_SIPS_no_noise
):
    (
        plan,
        solved,
    ) = dkg_astar_planner.generate_plan(dkg_state_three)
    observations = plan.states
    infered_goal = dkg_SIPS_no_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2"), ("has", "gem3")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem3")], 0.333, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem3")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_four_SIPS_no_noise(
    dkg_astar_planner, dkg_state_four, dkg_SIPS_no_noise
):
    (
        plan,
        solved,
    ) = dkg_astar_planner.generate_plan(dkg_state_four)
    observations = plan.states
    infered_goal = dkg_SIPS_no_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2"), ("has", "gem3")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem3")], 0.333, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem3")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_one_SIPS_noise(dkg_astar_planner, dkg_state_one, dkg_SIPS_noise):
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_one, r=1000, p=0.5)
    plan, solved = agent.execute_search()
    observations = apply_noise_to_states(plan.states, DKGObservation())
    infered_goal = dkg_SIPS_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 1.0, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_two_SIPS_noise(dkg_astar_planner, dkg_state_two, dkg_SIPS_noise):
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_two, r=1000, p=0.5)
    plan, solved = agent.execute_search()
    observations = apply_noise_to_states(plan.states, DKGObservation())
    infered_goal = dkg_SIPS_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.5, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.5, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_three_SIPS_noise(
    dkg_astar_planner, dkg_state_three, dkg_SIPS_noise
):
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_three, r=1000, p=0.5)
    (
        plan,
        solved,
    ) = agent.execute_search()
    observations = apply_noise_to_states(plan.states, DKGObservation())
    infered_goal = dkg_SIPS_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2"), ("has", "gem3")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem3")], 0.333, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem3")], 1.0, abs_tol=0.2
    )


def test_dkg_problem_four_SIPS_noise(dkg_astar_planner, dkg_state_four, dkg_SIPS_noise):
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_four, r=1000, p=0.5)
    plan, solved = agent.execute_search()
    observations = apply_noise_to_states(plan.states, DKGObservation())
    infered_goal = dkg_SIPS_noise.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2"), ("has", "gem3")],
        observations,
        resample_threshold=0.0,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.333, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem3")], 0.333, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem3")], 1.0, abs_tol=0.2
    )


def test_update_filter_single_agent(dkg_stoch_astar_planner, dkg_state_three):
    infer_model = SIPS(
        dkg_stoch_astar_planner,
        r=1000,
        p=0.5,
    )
    agent_one = BoundedRationalAgent(
        dkg_stoch_astar_planner, dkg_state_three, r=1, p=0.5
    )
    agent_one.step(5)
    while agent_one.action_step >= len(agent_one.curr_plan.actions):
        agent_one.replan()
    agent_two = BoundedRationalAgent(
        dkg_stoch_astar_planner, dkg_state_three, r=1, p=0.5
    )
    agent_two.step(5)
    while agent_two.action_step >= len(agent_two.curr_plan.actions):
        agent_two.replan()
    predicted_next_state_one = agent_one.curr_plan.states[agent_one.action_step + 1]
    predicted_next_state_two = agent_two.curr_plan.states[agent_two.action_step + 1]
    observation = predicted_next_state_one
    new_agents, new_weights = infer_model._update_filter(
        [agent_one, agent_two],
        torch.tensor([-0.6931, -0.6931]),
        1,
        [dkg_state_three, observation],
        dkg_state_three,
        [("has", "gem1"), ("has", "gem2"), ("has", "gem3")],
        0.0,
        False,
        0.0,
    )
    assert agent_one.curr_node.state == predicted_next_state_one
    assert agent_two.curr_node.state == predicted_next_state_two
    assert torch.isclose(new_weights[0], torch.tensor(-0.6931), atol=0.1)
    if predicted_next_state_one == predicted_next_state_two:
        assert torch.isclose(new_weights[1], torch.tensor(-0.6931), atol=0.1)
    else:
        assert torch.isneginf(new_weights[1])


def test_effective_sample_size_uniform(dkg_SIPS_no_noise):
    weights_uniform = torch.zeros(100)
    ess = dkg_SIPS_no_noise._effective_sample_size(weights_uniform)
    assert torch.isclose(ess, torch.tensor(100.0), atol=0.0001)


def test_effective_sample_size_peaked(dkg_SIPS_no_noise):
    weights_peaked = torch.ones(100) * -100
    weights_peaked[0] = 0.0
    ess = dkg_SIPS_no_noise._effective_sample_size(weights_peaked)
    assert torch.isclose(ess, torch.tensor(1.0), atol=0.01)


def test_resampling_new_weights_multinomial(dkg_SIPS_no_noise):
    weights = torch.rand(100)
    new_weights, new_indexes = dkg_SIPS_no_noise._multinomial_resample(weights)
    assert (new_weights == torch.zeros(100)).all()


def test_resampling_dead_trajs_not_resampled_multinomial(dkg_SIPS_no_noise):
    weights = torch.ones(100)
    weights[0:50] = -100.0
    new_weights, new_indexes = dkg_SIPS_no_noise._multinomial_resample(weights)
    assert (new_weights == torch.zeros(100)).all()
    assert torch.all(new_indexes >= 50)


def test_resampling_new_weights_residual(dkg_SIPS_no_noise):
    weights = torch.rand(100)
    new_weights, new_indexes = dkg_SIPS_no_noise._residual_resample(weights)
    assert (new_weights == torch.zeros(100)).all()


def test_resampling_uniform_weights_residual(dkg_SIPS_no_noise):
    weights = torch.ones(100)
    new_weights, new_indexes = dkg_SIPS_no_noise._residual_resample(weights)
    assert (new_weights == torch.zeros(100)).all()
    assert (new_indexes == torch.arange(100)).all()


def test_resampling_dead_trajs_not_resampled_residual(dkg_SIPS_no_noise):
    weights = torch.zeros(100)
    weights[0:50] = -100
    new_weights, new_indexes = dkg_SIPS_no_noise._residual_resample(weights)
    assert (new_weights == torch.zeros(100)).all()
    assert torch.all(new_indexes >= 50)


def test_goal_proposal(
    dkg_astar_planner,
    dkg_state_two,
    dkg_SIPS_no_noise,
):
    # Move Agent so that gem1 (goal gem) is closer
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_two, r=1000, p=0.5)
    agent.step(11)
    dist_gem1 = manhattan_gem_heuristic(
        agent.curr_node,
        ("has", "gem1"),
    )

    dist_gem2 = manhattan_gem_heuristic(
        agent.curr_node,
        ("has", "gem2"),
    )
    assert dist_gem1 < dist_gem2

    counts = {"gem1": 0, "gem2": 0, "gem3": 0}
    for _test in range(1000):
        alpha, new_agent = dkg_SIPS_no_noise._goal_rejuvenation(
            agent,
            dkg_state_two,
            [("has", "gem1"), ("has", "gem2")],
        )
        counts[new_agent.curr_node.state.goal[1]] += 1
    # Test that Rejuvenated goals are more commonly gem1 than gem2
    assert counts["gem1"] > counts["gem2"]


def test_problem_one_resample_and_rejuvenate(
    dkg_state_one, dkg_stoch_astar_planner, dkg_astar_planner
):
    model = SIPS(dkg_stoch_astar_planner, DKGObservation())
    (
        plan,
        solved,
    ) = dkg_astar_planner.generate_plan(dkg_state_one)
    observations = plan.states
    infered_goal = model.infer(
        100,
        plan.states[0],
        [("has", "gem1")],
        observations,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 1.0, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_problem_two_resample_and_rejuvenate(
    dkg_state_two, dkg_stoch_astar_planner, dkg_astar_planner
):
    model = SIPS(dkg_stoch_astar_planner, DKGObservation())
    (
        plan,
        solved,
    ) = dkg_astar_planner.generate_plan(dkg_state_two)
    observations = plan.states
    infered_goal = model.infer(
        100,
        plan.states[0],
        [("has", "gem1"), ("has", "gem2")],
        observations,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.5, abs_tol=0.2)
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.5, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_cgem_problem_one():
    state = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-1.json"
    )
    domain = CGemDomain()
    planner = StochasticAstarPlanner(domain, manhattan_gem_heuristic, 0.1)
    plan, solved = planner.generate_plan(state)
    model = SIPS(planner, r=8, p=0.95, noise_model=CGemObservation())
    goals = [("has", "gem1"), ("has", "gem2")]
    infered_goal = model.infer(
        600,
        state,
        goals,
        plan.states,
        resample_threshold=0.25,
        rejuvenate=False,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem1")], 0.5, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem1")], 1.0, abs_tol=0.2
    )


def test_cgem_problem_two():

    state = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-2.json"
    )
    domain = CGemDomain()
    planner = StochasticAstarPlanner(domain, manhattan_gem_heuristic, 0.1)
    state = dataclasses.replace(state, goal=("has", "gem2"))
    plan, solved = planner.generate_plan(state)
    model = SIPS(planner, r=8, p=0.95, noise_model=CGemObservation())
    goals = [("has", "gem1"), ("has", "gem2")]
    infered_goal = model.infer(
        600,
        state,
        goals,
        plan.states,
        resample_threshold=0.25,
        rejuvenate=False,
        goal_rejuvenation_prob=1.0,
    )
    assert math.isclose(infered_goal[0][("has", "gem2")], 0.5, abs_tol=0.2)
    assert math.isclose(
        get_last_inference_prediction(infered_goal)[("has", "gem2")], 1.0, abs_tol=0.2
    )
