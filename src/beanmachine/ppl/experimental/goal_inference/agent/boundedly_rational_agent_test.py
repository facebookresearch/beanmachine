# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import dataclasses

import pytest
from beanmachine.facebook.goal_inference.agent.boundedly_rational_agent import (
    BoundedRationalAgent,
)
from beanmachine.facebook.goal_inference.planner.planner import (
    get_plan_from_actions,
    Plan,
)


@pytest.fixture
def agent_one(dkg_stoch_astar_planner, dkg_state_one):
    return BoundedRationalAgent(dkg_stoch_astar_planner, dkg_state_one, r=10, p=0.5)


@pytest.fixture
def agent_two(dkg_stoch_astar_planner, dkg_state_two):
    return BoundedRationalAgent(dkg_stoch_astar_planner, dkg_state_two, r=10, p=0.5)


@pytest.fixture
def agent_three(dkg_stoch_astar_planner, dkg_state_three):
    return BoundedRationalAgent(dkg_stoch_astar_planner, dkg_state_three, r=10, p=0.5)


def test_initialize_dkg_bounded_agent(agent_one):
    assert agent_one.agent_step == 0


def test_dkg_bounded_agent_problem_1(agent_one):
    plan, solved = agent_one.execute_search()
    assert solved
    planner = agent_one.planner
    state = agent_one.curr_node.state
    assert planner.domain.evaluate_goal(state)


def test_dkg_bounded_agent_problem_2(agent_two):
    plan, solved = agent_two.execute_search()
    assert solved
    planner = agent_two.planner
    state = agent_two.curr_node.state
    assert planner.domain.evaluate_goal(state)


def test_dkg_bounded_agent_impossible_task(agent_two):
    # Make task impossible
    new_at = agent_two.curr_node.state.at
    new_at.pop("key1")
    agent_two.curr_node.state = dataclasses.replace(
        agent_two.curr_node.state, keys={}, at=new_at
    )
    plan, solved = agent_two.execute_search()
    assert (agent_two.agent_step >= 1000) and not solved


def test_dkg_bounded_agent_replan(agent_two):
    agent_two.replan()
    curr_plan = agent_two.curr_plan
    assert isinstance(agent_two.curr_plan, Plan)
    assert len(curr_plan.states) == len(curr_plan.actions) + 1


def test_replan_after_solved_is_unchanged(agent_two):
    plan, solved = agent_two.execute_search()
    num_plans = len(agent_two.plan_history)
    assert solved
    agent_two.replan()
    curr_plan = agent_two.curr_plan
    assert curr_plan.states == agent_two.plan_history[num_plans - 1].states


def test_dkg_bounded_agent_execute_plan(agent_two):
    agent_two.replan()
    curr_plan = agent_two.curr_plan
    initial_steps = agent_two.agent_step
    agent_two.execute_plan()
    assert agent_two.curr_node.state == curr_plan.states[-1]
    assert agent_two.agent_step == initial_steps + len(curr_plan.actions)


def test_dkg_bounded_agent_execute_plan_break(agent_two):
    agent_two.replan()
    while len(agent_two.curr_plan.actions) < 3:
        agent_two.replan()
    curr_plan = agent_two.curr_plan
    initial_steps = agent_two.agent_step
    # Create an unexpected state
    curr_plan.states[-1] = curr_plan.states[0]
    agent_two.execute_plan()
    assert agent_two.curr_node.state == curr_plan.states[-2]
    assert agent_two.agent_step == initial_steps + len(curr_plan.actions) - 1


def test_dkg_bounded_agent_total_plan(agent_two):
    plan, solved = agent_two.execute_search()
    assert solved
    assert isinstance(plan, Plan)
    assert len(plan.states) == len(plan.actions) + 1
    assert len(plan.actions) == agent_two.agent_step


def test_execute_step_match_expected(agent_two):
    agent_two.replan()
    while len(agent_two.curr_plan.actions) < 3:
        agent_two.replan()
    curr_plan = agent_two.curr_plan
    agent_two._execute_step()
    assert agent_two.agent_step == 1
    assert agent_two.action_step == 1
    assert agent_two.curr_node.state == curr_plan.states[1]
    agent_two._execute_step()
    assert agent_two.agent_step == 2
    assert agent_two.action_step == 2
    assert agent_two.curr_node.state == curr_plan.states[2]


def test_execute_step_break_expected(agent_two):
    agent_two.replan()
    while len(agent_two.curr_plan.actions) < 3:
        agent_two.replan()
    curr_plan = agent_two.curr_plan
    # Create an unexpected state
    curr_plan.states[2] = curr_plan.states[0]
    agent_two._execute_step()
    assert agent_two.agent_step == 1
    assert agent_two.action_step == 1
    assert agent_two.curr_node.state == curr_plan.states[1]
    agent_two._execute_step()
    assert agent_two.agent_step == 1
    assert agent_two.action_step == len(curr_plan.actions)
    assert agent_two.curr_node.state == curr_plan.states[1]


def test_advance_single_time_step(agent_three):
    agent_three.replan()
    while len(agent_three.curr_plan.actions) < 3:
        agent_three.replan()
    agent_three.step()
    curr_plan = agent_three.curr_plan
    assert agent_three.agent_step == 1
    assert agent_three.action_step == 1
    assert agent_three.curr_node.state == curr_plan.states[1]
    agent_three.step()
    assert agent_three.agent_step == 2
    assert agent_three.action_step == 2
    assert agent_three.curr_node.state == curr_plan.states[2]


def test_advance_n_time_steps(agent_three):
    agent_three.replan()
    while len(agent_three.curr_plan.actions) < 3:
        agent_three.replan()
    agent_three.step()
    curr_plan = agent_three.curr_plan
    assert agent_three.agent_step == 1
    assert agent_three.action_step == 1
    assert agent_three.curr_node.state == curr_plan.states[1]
    # Time step through a new plan
    steps_for_new_plan = len(curr_plan.actions)
    agent_three.step(steps_for_new_plan)
    curr_plan = agent_three.curr_plan
    assert agent_three.agent_step == steps_for_new_plan + 1
    assert agent_three.action_step == 1
    assert agent_three.curr_node.state == curr_plan.states[1]


def test_advance_n_time_steps_break_expected(
    agent_three,
):
    agent_three.replan()
    while len(agent_three.curr_plan.actions) < 3:
        agent_three.replan()
    curr_plan = agent_three.curr_plan
    # Create an unexpected state
    curr_plan.states[-1] = curr_plan.states[0]
    # Should break from first plan one step early and be one step into second plan
    agent_three.step(len(curr_plan.actions))
    assert agent_three.agent_step == len(curr_plan.actions)
    assert agent_three.action_step == 1
    assert agent_three.curr_node.state == agent_three.curr_plan.states[1]


def test_advance_time_steps_after_solution(agent_two):
    plan, solved = agent_two.execute_search()
    assert solved
    old_time_step = agent_two.agent_step
    old_state = agent_two.curr_node.state
    agent_two.step()
    assert agent_two.agent_step == old_time_step
    assert agent_two.curr_node.state == old_state


def test_agent_copy(agent_two):
    new_agent = copy.copy(agent_two)
    assert new_agent.agent_step == agent_two.agent_step
    new_agent.step()
    assert new_agent.agent_step != agent_two.agent_step


def test_bounded_agent_set_time(dkg_stoch_astar_planner, dkg_state_three):
    agent = BoundedRationalAgent(dkg_stoch_astar_planner, dkg_state_three, r=2, p=0.95)
    agent_actions = [
        ["right"],
        ["right"],
        ["right"],
    ]
    plan = get_plan_from_actions(
        dkg_stoch_astar_planner.domain, dkg_state_three, agent_actions
    )
    agent.plan_history.append(plan)
    agent.plan_index += 1
    length_plan_one = len(agent.curr_plan.actions)
    agent.execute_plan()
    agent.replan()
    agent.execute_plan()
    agent.replan()
    agent.execute_plan()
    time = length_plan_one + 1
    agent._set_to_timestep(time)
    assert agent.plan_index == 1
    assert agent.agent_step == time
