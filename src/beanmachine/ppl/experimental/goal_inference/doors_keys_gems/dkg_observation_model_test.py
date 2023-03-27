# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses

import math

import torch

from beanmachine.facebook.goal_inference.doors_keys_gems.dkg_observation_model import (
    DKGObservation,
)
from torch.distributions.normal import Normal


def test_dkg_gaussian_noise(dkg_state_two):
    noise_model_bern = DKGObservation(gaussian_noise=0.001, bernoulli_noise=1.0)
    new_state = noise_model_bern.apply_noise(dkg_state_two)
    assert not new_state.at.keys()
    assert not new_state.has.keys()
    assert not new_state.doors
    assert math.isclose(new_state.x, dkg_state_two.x, abs_tol=0.1)
    assert math.isclose(new_state.y, dkg_state_two.y, abs_tol=0.1)

    noise_model_gaus = DKGObservation(gaussian_noise=0.25, bernoulli_noise=0.0)
    new_state = noise_model_gaus.apply_noise(dkg_state_two)
    assert new_state.at.keys() == dkg_state_two.at.keys()
    assert new_state.has.keys() == dkg_state_two.has.keys()
    assert new_state.doors == dkg_state_two.doors
    assert new_state.x != dkg_state_two.x
    assert new_state.y != dkg_state_two.y

    noise_model_mixed = DKGObservation(gaussian_noise=0.25, bernoulli_noise=0.05)
    new_state_two = dataclasses.replace(dkg_state_two, x=dkg_state_two.x + 1)
    new_prob = Normal(torch.tensor(dkg_state_two.x), torch.tensor(0.25)).log_prob(
        torch.tensor(dkg_state_two.x + 1)
    )

    new_prob += Normal(torch.tensor(dkg_state_two.y), torch.tensor(0.25)).log_prob(
        torch.tensor(dkg_state_two.y)
    )

    new_prob += torch.log(torch.tensor(0.95)) * (
        len(dkg_state_two.doors) + len(dkg_state_two.items.keys())
    )
    assert torch.isclose(
        noise_model_mixed.get_log_prob(dkg_state_two, new_state_two),
        new_prob,
        atol=0.01,
    )
