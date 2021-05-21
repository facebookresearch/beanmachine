import beanmachine.ppl as bm
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


world = SimpleWorld()
world.call(bar())
hmc = HMCProposer(world, trajectory_length=1.0)


def test_potential_grads():
    pe, pe_grad = hmc._potential_grads(world)
    assert isinstance(pe, torch.Tensor)
    assert pe.numel() == 1
    for node in world.latent_nodes:
        assert node in pe_grad
        assert isinstance(pe_grad[node], torch.Tensor)
        assert pe_grad[node].shape == world[node].shape


def test_initialize_momentums():
    momentums = hmc._initialize_momentums(world)
    for node in world.latent_nodes:
        assert node in momentums
        assert isinstance(momentums[node], torch.Tensor)
        assert momentums[node].shape == world[node].shape


def test_kinetic_grads():
    momentums = hmc._initialize_momentums(world)
    ke = hmc._kinetic_energy(momentums)
    assert isinstance(ke, torch.Tensor)
    assert ke.numel() == 1
    ke_grad = hmc._kinetic_grads(momentums)
    for node in world.latent_nodes:
        assert node in ke_grad
        assert isinstance(ke_grad[node], torch.Tensor)
        assert ke_grad[node].shape == world[node].shape


def test_leapfrog_step():
    step_size = 0.0
    momentums = hmc._initialize_momentums(world)
    new_world, new_momentums, pe, pe_grad = hmc._leapfrog_step(
        world, momentums, step_size
    )
    assert momentums == new_momentums
    assert new_world._transformed_values == world._transformed_values
