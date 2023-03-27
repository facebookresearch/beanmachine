# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from beanmachine.facebook.goal_inference.agent.observation_model import (
    DeterministicObservation,
)
from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_observation_model import (
    DKGObservation,
)

from beanmachine.facebook.goal_inference.planner.planner import (
    get_plan_from_actions,
    StateNode,
)

from beanmachine.facebook.goal_inference.planner.stoch_astar import (
    StochasticAstarPlanner,
    StochasticAstarProposalPlanner,
)
from beanmachine.facebook.goal_inference.utils import manhattan_gem_heuristic


def test_stoch_astar_generate_plan_dkg_problem_1(dkg_domain, dkg_state_one):
    st_astar_solver = StochasticAstarPlanner(dkg_domain, manhattan_gem_heuristic, 0.5)
    plan, solved = st_astar_solver.generate_plan(dkg_state_one)
    assert solved
    assert plan.actions[-1] == ["pickup", "gem1"]


def test_stoch_astar_generate_plan_dkg_problem_2(dkg_domain, dkg_state_two):
    st_astar_solver = StochasticAstarPlanner(dkg_domain, manhattan_gem_heuristic, 0.5)
    plan, solved = st_astar_solver.generate_plan(dkg_state_two)
    assert solved
    assert plan.actions[-1] == ["pickup", "gem1"]


def test_path_propose(dkg_domain, dkg_state_three, dkg_observation_set):
    planner = StochasticAstarPlanner(dkg_domain, manhattan_gem_heuristic, 0.1)
    proposal_planner = StochasticAstarProposalPlanner(planner)
    new_plan, solved = proposal_planner.propose(
        dkg_state_three, dkg_observation_set, 1000, DeterministicObservation()
    )
    for time_step in range(10):
        assert new_plan.states[time_step] == dkg_observation_set[time_step]


def test_get_log_prob(dkg_domain, dkg_state_three, dkg_observation_set):
    noise_model = DKGObservation()
    planner = StochasticAstarPlanner(dkg_domain, manhattan_gem_heuristic, 0.1)
    proposal_planner = StochasticAstarProposalPlanner(planner)
    new_actions = [
        ["right"],
        ["right"],
        ["right"],
    ]
    new_plan = get_plan_from_actions(dkg_domain, dkg_state_three, new_actions)
    new_plan.budget = 4
    new_plan.visited_nodes = [
        StateNode(new_plan.states[0], None, []),
        StateNode(new_plan.states[1], None, []),
        StateNode(new_plan.states[2], None, []),
        StateNode(new_plan.states[3], None, []),
    ]

    computed_prob = torch.tensor(0.0)

    # Step 1

    # No Option Moves Right

    # Step 2

    # Options RR or RL

    node_weights = torch.tensor([-5.0 / 0.1, -7.0 / 0.1])

    # Bias

    # state RR
    # state RL
    state_rr = dkg_domain.execute(new_plan.states[1], "right")
    state_rl = dkg_domain.execute(new_plan.states[1], "left")
    node_weights[0] += noise_model.get_log_prob(state_rr, dkg_observation_set[2])
    node_weights[1] += noise_model.get_log_prob(state_rl, dkg_observation_set[2])

    node_weights -= torch.logsumexp(node_weights, dim=0)
    computed_prob += node_weights[0]

    # Step 3

    # Options RRR RRL RL

    node_weights = torch.tensor([-4.0 / 0.1, -6.0 / 0.1, -7.0 / 0.1])

    # Bias
    state_rrr = dkg_domain.execute(new_plan.states[2], "right")
    state_rrl = dkg_domain.execute(new_plan.states[2], "left")
    state_rl = dkg_domain.execute(new_plan.states[1], "left")
    node_weights[0] += noise_model.get_log_prob(state_rrr, dkg_observation_set[3])
    node_weights[1] += noise_model.get_log_prob(state_rrl, dkg_observation_set[3])
    node_weights[2] += noise_model.get_log_prob(state_rl, dkg_observation_set[2])

    node_weights -= torch.logsumexp(node_weights, dim=0)
    computed_prob += node_weights[0]

    assert torch.isclose(
        computed_prob,
        proposal_planner.get_log_prob(new_plan, dkg_observation_set, noise_model),
        atol=0.1,
    )
