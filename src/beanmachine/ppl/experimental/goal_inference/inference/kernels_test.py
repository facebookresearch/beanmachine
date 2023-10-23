# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math

import pytest

import torch

from beanmachine.facebook.goal_inference.agent.boundedly_rational_agent import (
    BoundedRationalAgent,
)

from beanmachine.facebook.goal_inference.agent.observation_model import (
    DeterministicObservation,
)

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_observation_model import (
    DKGObservation,
)
from beanmachine.facebook.goal_inference.inference.kernels import (
    DeviateTimeRejuvenateProposal,
    HeuristicGoalProposal,
    PathRejuvenateProposal,
    StratifiedGoalPrior,
    UniformGoalPrior,
)

from beanmachine.facebook.goal_inference.planner.planner import (
    get_execution_path,
    get_plan_from_actions,
)
from beanmachine.facebook.goal_inference.utils import manhattan_gem_heuristic


@pytest.fixture(scope="function")
def dkg_agent(dkg_state_three, dkg_astar_planner):
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_three)
    agent_actions = [
        ["right"],
        ["right"],
        ["right"],
        ["up"],
        ["up"],
        ["up"],
        ["up"],
        ["left"],
        ["down"],
        ["down"],
        ["right"],
        ["down"],
    ]
    plan = get_plan_from_actions(
        dkg_astar_planner.domain, dkg_state_three, agent_actions
    )
    agent.plan_history.append(plan)
    agent.plan_index += 1
    agent.execute_plan()
    return agent


def test_time_proposal_distribution_noise(dkg_observation_set, dkg_agent):
    proposal = DeviateTimeRejuvenateProposal()
    noise_model = DKGObservation()
    prob_dist = proposal._get_prob(dkg_agent, dkg_observation_set, noise_model)
    assert prob_dist.shape[0] == 12
    assert torch.isclose(torch.argmax(prob_dist), torch.tensor(5))
    assert torch.isclose(prob_dist[5], torch.tensor(10.0 / 21.0))
    assert torch.isclose(prob_dist[0], torch.tensor(1.0 / 21.0))
    assert torch.isclose(prob_dist[-1], torch.tensor(1.0 / 21.0))


def test_time_proposal_distribution_no_noise(dkg_observation_set, dkg_agent):
    proposal = DeviateTimeRejuvenateProposal()
    noise_model = DeterministicObservation()
    prob_dist = proposal._get_prob(dkg_agent, dkg_observation_set, noise_model)
    assert prob_dist.shape[0] == 12
    assert torch.isclose(torch.argmax(prob_dist), torch.tensor(5))
    assert torch.isclose(prob_dist[5], torch.tensor(10.0 / 21.0))
    assert torch.isclose(prob_dist[0], torch.tensor(1.0 / 21.0))
    assert torch.isclose(prob_dist[-1], torch.tensor(1.0 / 21.0))


def test_time_proposal_log_prob_noise(dkg_observation_set, dkg_agent):
    proposal = DeviateTimeRejuvenateProposal()
    noise_model = DKGObservation()
    prob_max = proposal.get_log_prob(dkg_agent, dkg_observation_set, noise_model, 5)
    prob_min = proposal.get_log_prob(dkg_agent, dkg_observation_set, noise_model, 1)
    assert torch.isclose(prob_max, torch.log(torch.tensor(10.0 / 21.0)))
    assert torch.isclose(prob_min, torch.log(torch.tensor(1.0 / 21.0)))


def test_time_proposal_log_prob_no_noise(dkg_observation_set, dkg_agent):
    proposal = DeviateTimeRejuvenateProposal()
    noise_model = DeterministicObservation()
    prob_max = proposal.get_log_prob(dkg_agent, dkg_observation_set, noise_model, 5)
    prob_min = proposal.get_log_prob(dkg_agent, dkg_observation_set, noise_model, 1)
    assert torch.isclose(prob_max, torch.log(torch.tensor(10.0 / 21.0)))
    assert torch.isclose(prob_min, torch.log(torch.tensor(1.0 / 21.0)))


def test_time_proposal_propose(dkg_observation_set, dkg_agent):
    proposal = DeviateTimeRejuvenateProposal()
    noise_model = DKGObservation()
    proposed_time, log_prob_proposed_time = proposal.propose(
        dkg_agent, dkg_observation_set, noise_model
    )
    assert proposed_time < len(dkg_observation_set)
    assert torch.isclose(
        log_prob_proposed_time,
        torch.log(
            proposal._get_prob(dkg_agent, dkg_observation_set, noise_model)[
                proposed_time
            ]
        ),
    )


def test_path_propose(dkg_astar_planner, dkg_state_three, dkg_observation_set):
    agent = BoundedRationalAgent(dkg_astar_planner, dkg_state_three, r=1, p=0.9999)
    agent_actions = [
        ["right"],
        ["right"],
        ["right"],
        ["up"],
        ["up"],
    ]
    plan = get_plan_from_actions(
        dkg_astar_planner.domain, dkg_state_three, agent_actions
    )
    agent.plan_history.append(plan)
    agent.plan_index += 1
    agent.execute_plan()
    agent_actions = [
        ["up"],
        ["up"],
        ["left"],
        ["down"],
        ["down"],
        ["right"],
        ["down"],
    ]
    plan = get_plan_from_actions(
        dkg_astar_planner.domain, agent.curr_node.state, agent_actions
    )
    agent.action_step = 0
    agent.plan_history.append(plan)
    agent.plan_index += 1
    agent.execute_plan()

    proposal = PathRejuvenateProposal()
    noise_model = DeterministicObservation()
    new_agent, prob = proposal.propose(agent, dkg_observation_set[:13], 5, noise_model)
    new_path = get_execution_path(new_agent.curr_node)[0]
    assert new_path == dkg_observation_set[:13]
    assert prob < 0.0


def test_uniform_prior_goal_prob():
    prior = UniformGoalPrior()
    goal_record = {("goal1"): 0, ("goal2"): 0, ("goal3"): 0}
    for _ in range(100):
        new_goal = prior.sample_prior(list(goal_record.keys()))
        goal_record[new_goal] += 1
    assert math.isclose(goal_record[("goal1")], 33, abs_tol=10)
    assert math.isclose(goal_record[("goal1")], 33, abs_tol=10)
    assert math.isclose(goal_record[("goal1")], 33, abs_tol=10)


def test_heuristic_goal_proposal(dkg_agent):
    goals = [("has", "gem1"), ("has", "gem2"), ("has", "gem3")]
    proposal = HeuristicGoalProposal(heuristic=dkg_agent.planner.heuristic)
    prop_dist = proposal._get_goal_log_probs(dkg_agent, goals)

    dist_goal_1 = float(manhattan_gem_heuristic(dkg_agent.curr_node, ("has", "gem1")))
    dist_goal_2 = float(manhattan_gem_heuristic(dkg_agent.curr_node, ("has", "gem2")))
    dist_goal_3 = float(manhattan_gem_heuristic(dkg_agent.curr_node, ("has", "gem3")))

    log_probs = torch.tensor([dist_goal_1, dist_goal_2, dist_goal_3])
    log_probs = -log_probs / 10.0
    log_probs -= torch.logsumexp(log_probs, dim=0)

    assert torch.isclose(log_probs[0], prop_dist[0], atol=0.1)
    assert torch.isclose(log_probs[1], prop_dist[1], atol=0.1)
    assert torch.isclose(log_probs[2], prop_dist[2], atol=0.1)

    assert torch.isclose(
        prop_dist[0], proposal.get_log_prob(dkg_agent, goals, ("has", "gem1")), atol=0.1
    )
    assert torch.isclose(
        prop_dist[1], proposal.get_log_prob(dkg_agent, goals, ("has", "gem2")), atol=0.1
    )
    assert torch.isclose(
        prop_dist[2], proposal.get_log_prob(dkg_agent, goals, ("has", "gem3")), atol=0.1
    )


def test_stratified_prior_goal_prob():
    prior = StratifiedGoalPrior(100)
    goal_record = {("goal1"): 0, ("goal2"): 0, ("goal3"): 0}
    for _ in range(100):
        new_goal = prior.sample_prior(list(goal_record.keys()))
        goal_record[new_goal] += 1
    assert math.isclose(goal_record[("goal1")], 33, abs_tol=2)
    assert math.isclose(goal_record[("goal1")], 33, abs_tol=2)
    assert math.isclose(goal_record[("goal1")], 33, abs_tol=2)
