# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
import torch
from beanmachine.ppl.examples.hierarchical_models import UniformNormalModel
from beanmachine.ppl.inference.proposer.hmc_proposer import HMCProposer
from beanmachine.ppl.world import World
from torch import tensor

from ...utils.fixtures import approx_all, parametrize_inference, parametrize_model


pytestmark = parametrize_model(
    [UniformNormalModel(tensor(0.0), tensor(1.0), tensor(1.0))]
)


@pytest.fixture
def world(model):
    w = World()
    w.call(model.obs())
    return w


@pytest.fixture
def hmc(world):
    hmc_proposer = HMCProposer(world, world.latent_nodes, 10, trajectory_length=1.0)
    return hmc_proposer


def test_potential_grads(hmc):
    pe, pe_grad = hmc._potential_grads(hmc._positions)
    assert isinstance(pe, torch.Tensor)
    assert pe.numel() == 1
    assert isinstance(pe_grad, torch.Tensor)
    assert pe_grad.shape == hmc._positions.shape


def test_kinetic_grads(hmc):
    momentums = hmc._initialize_momentums(hmc._positions)
    ke = hmc._kinetic_energy(momentums, hmc._mass_inv)
    assert isinstance(ke, torch.Tensor)
    assert ke.numel() == 1
    ke_grad = hmc._kinetic_grads(momentums, hmc._mass_inv)
    assert isinstance(ke_grad, torch.Tensor)
    assert ke_grad.shape == hmc._positions.shape


def test_leapfrog_step(hmc):
    step_size = torch.tensor(0.0)
    momentums = hmc._initialize_momentums(hmc._positions)
    new_positions, new_momentums, pe, pe_grad = hmc._leapfrog_step(
        hmc._positions, momentums, step_size, hmc._mass_inv
    )
    assert approx_all(momentums, new_momentums)
    assert approx_all(hmc._positions, new_positions)


@parametrize_inference(
    # forcing the step_size to be 0 for HMC/ NUTS
    [
        bm.GlobalNoUTurnSampler(initial_step_size=0.0),
        bm.GlobalHamiltonianMonteCarlo(trajectory_length=1.0, initial_step_size=0.0),
    ],
)
def test_step_size_exception(model, inference):
    queries = [model.mean()]
    observations = {model.obs(): torch.tensor(0.5)}

    with pytest.raises(ValueError):
        inference.infer(
            queries,
            observations,
            num_samples=20,
            num_chains=1,
        )
