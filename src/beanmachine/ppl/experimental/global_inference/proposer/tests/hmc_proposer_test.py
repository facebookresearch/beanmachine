import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.proposer.hmc_proposer import (
    HMCProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


@bm.random_variable
def foo():
    return dist.Uniform(0.0, 1.0)


@bm.random_variable
def bar():
    return dist.Normal(foo(), 1.0)


@pytest.fixture
def world():
    w = SimpleWorld()
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
    for node, z in hmc._positions.items():
        assert node in pe_grad
        assert isinstance(pe_grad[node], torch.Tensor)
        assert pe_grad[node].shape == z.shape


def test_kinetic_grads(hmc):
    momentums = hmc._initialize_momentums(hmc._positions)
    ke = hmc._kinetic_energy(momentums, hmc._mass_inv)
    assert isinstance(ke, torch.Tensor)
    assert ke.numel() == 1
    ke_grad = hmc._kinetic_grads(momentums, hmc._mass_inv)
    for node, z in hmc._positions.items():
        assert node in ke_grad
        assert isinstance(ke_grad[node], torch.Tensor)
        assert len(ke_grad[node]) == z.numel()


def test_leapfrog_step(hmc):
    step_size = 0.0
    momentums = hmc._initialize_momentums(hmc._positions)
    new_positions, new_momentums, pe, pe_grad = hmc._leapfrog_step(
        hmc._positions, momentums, step_size, hmc._mass_inv
    )
    assert momentums == new_momentums
    assert new_positions == hmc._positions
