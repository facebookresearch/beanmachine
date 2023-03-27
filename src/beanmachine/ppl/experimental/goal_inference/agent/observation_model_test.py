# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import dataclasses

import torch

from beanmachine.facebook.goal_inference.agent.observation_model import (
    DeterministicObservation,
)


def test_deterministic_noise(dkg_state_two):
    noise_model = DeterministicObservation()
    new_state = noise_model.apply_noise(dkg_state_two)
    assert new_state == dkg_state_two
    new_state_two = dataclasses.replace(dkg_state_two, x=dkg_state_two.x + 1)
    assert torch.isneginf(noise_model.get_log_prob(new_state_two, dkg_state_two))
    assert noise_model.get_log_prob(new_state, dkg_state_two) == 0.0
