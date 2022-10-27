# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.hmc_proposer import HMCProposer
from beanmachine.ppl.world import World


@bm.random_variable
def foo():
    return dist.Uniform(0.0, 1.0)


@bm.random_variable
def bar():
    return dist.Normal(foo(), 1.0)


@pytest.fixture
def world():
    w = World()
    w.call(bar())
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
    assert torch.allclose(momentums, new_momentums)
    assert torch.allclose(hmc._positions, new_positions)


@pytest.mark.parametrize(
    # forcing the step_size to be 0 for HMC/ NUTS
    "algorithm",
    [
        bm.GlobalNoUTurnSampler(initial_step_size=0.0),
        bm.GlobalHamiltonianMonteCarlo(trajectory_length=1.0, initial_step_size=0.0),
    ],
)
def test_step_size_exception(algorithm):
    queries = [foo()]
    observations = {bar(): torch.tensor(0.5)}

    with pytest.raises(ValueError):
        algorithm.infer(
            queries,
            observations,
            num_samples=20,
            num_chains=1,
        )
