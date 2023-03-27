# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses
import math

import torch
from beanmachine.facebook.goal_inference.continuous_gems.cgem_observation_model import (
    CGemObservation,
)
from beanmachine.facebook.goal_inference.continuous_gems.cgem_parse import parse
from torch.distributions.normal import Normal


def test_cgem_observation_noise_positions_and_angles():
    state = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-2.json"
    )
    noise_model = CGemObservation(bernoulli_noise=0.0)
    noisy_state = noise_model.apply_noise(state)
    assert state.x != noisy_state.x
    assert state.y != noisy_state.y
    assert state.angle != noisy_state.angle
    assert state.has == noisy_state.has
    assert state.at == noisy_state.at


def test_cgem_observation_noise_gems():
    state = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-2.json"
    )
    noise_model = CGemObservation(
        position_noise=0.01, angle_noise=0.01, bernoulli_noise=1.0
    )
    noisy_state = noise_model.apply_noise(state)
    assert math.isclose(state.x, noisy_state.x, abs_tol=0.25)
    assert math.isclose(state.y, noisy_state.y, abs_tol=0.25)
    assert not noisy_state.at.keys()
    assert not noisy_state.has.keys()


def test_cgem_observation_log_prob():
    state = parse(
        "beanmachine/facebook/goal_inference/continuous_gems/test_problems/problem-2.json"
    )
    noise_model = CGemObservation()
    new_state = dataclasses.replace(state, x=state.x + 1.0)
    log_prob = noise_model.get_log_prob(state, new_state)

    calc_log_prob = torch.tensor(0.0)
    calc_log_prob += Normal(torch.tensor(state.x), torch.tensor(0.5)).log_prob(
        torch.tensor(state.x + 1)
    )

    calc_log_prob += Normal(torch.tensor(state.y), torch.tensor(0.5)).log_prob(
        torch.tensor(state.y)
    )

    calc_log_prob += Normal(torch.tensor(state.angle), torch.tensor(120.0)).log_prob(
        torch.tensor(state.angle)
    )

    calc_log_prob += torch.log(torch.tensor(0.95)) * (len(state.gems.keys()))

    assert torch.isclose(calc_log_prob, log_prob, atol=0.01)
