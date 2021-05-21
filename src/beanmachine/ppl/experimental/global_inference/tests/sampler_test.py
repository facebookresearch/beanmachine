import beanmachine.ppl as bm
import torch
import torch.distributions as dist


class SampleModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(0.0, 1.0)

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), 1.0)


def test_sampler():
    model = SampleModel()
    nuts = bm.GlobalNoUTurnSampler()
    queries = [model.foo()]
    observations = {model.bar(): torch.tensor(0.5)}
    num_samples = 10
    sampler = nuts.sampler(queries, observations, num_samples)
    worlds = list(sampler)
    assert len(worlds) == num_samples
    for world in worlds:
        assert model.foo() in world
        with world:
            assert isinstance(model.foo(), torch.Tensor)


def test_two_samplers():
    model = SampleModel()
    queries = [model.foo()]
    observations = {model.bar(): torch.tensor(0.5)}
    nuts_sampler = bm.GlobalNoUTurnSampler().sampler(queries, observations)
    hmc_sampler = bm.GlobalHamiltonianMonteCarlo(1.0).sampler(queries, observations)
    world = next(nuts_sampler)
    # it's possible to use multiple sampler interchangably to update the worlds (or
    # in general, pass a new world to sampler and continue inference with existing
    # hyperparameters)
    for _ in range(3):
        world = hmc_sampler.send(world)
        world = nuts_sampler.send(world)
    assert model.foo() in world
    assert model.bar() in world
