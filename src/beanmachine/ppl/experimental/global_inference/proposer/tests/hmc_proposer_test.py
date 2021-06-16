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
    hmc_proposer = HMCProposer(world, trajectory_length=1.0)
    return hmc_proposer


def test_potential_grads(world, hmc):
    pe, pe_grad = hmc._potential_grads(world)
    assert isinstance(pe, torch.Tensor)
    assert pe.numel() == 1
    for node in world.latent_nodes:
        assert node in pe_grad
        assert isinstance(pe_grad[node], torch.Tensor)
        assert pe_grad[node].shape == world.get_transformed(node).shape


def test_kinetic_grads(world, hmc):
    momentums = hmc._initialize_momentums(world)
    ke = hmc._kinetic_energy(momentums, hmc._mass_inv)
    assert isinstance(ke, torch.Tensor)
    assert ke.numel() == 1
    ke_grad = hmc._kinetic_grads(momentums, hmc._mass_inv)
    for node in world.latent_nodes:
        assert node in ke_grad
        assert isinstance(ke_grad[node], torch.Tensor)
        assert len(ke_grad[node]) == world.get_transformed(node).numel()


def test_leapfrog_step(world, hmc):
    step_size = 0.0
    momentums = hmc._initialize_momentums(world)
    new_world, new_momentums, pe, pe_grad = hmc._leapfrog_step(
        world, momentums, step_size, hmc._mass_inv
    )
    assert momentums == new_momentums
    assert new_world._transformed_values == world._transformed_values
