import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.proposer.base_proposer import (
    BaseProposer,
)
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


class SampleModel:
    @bm.random_variable
    def foo(self):
        return dist.Normal(0.0, 1.0)

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), 1.0)

    @bm.functional
    def baz(self):
        return self.bar() * 2.0


def test_inference():
    model = SampleModel()
    nuts = bm.GlobalNoUTurnSampler()
    queries = [model.foo(), model.baz()]
    observations = {model.bar(): torch.tensor(0.5)}
    num_samples = 30
    num_chains = 2
    samples = nuts.infer(
        queries,
        observations,
        num_samples,
        num_adaptive_samples=num_samples,
        num_chains=num_chains,
    )

    assert model.foo() in samples
    assert isinstance(samples[model.foo()], torch.Tensor)
    assert samples[model.foo()].shape == (num_chains, num_samples)
    assert samples.get_num_samples(include_adapt_steps=True) == num_samples * 2


def test_get_proposers():
    world = SimpleWorld()
    model = SampleModel()
    world.call(model.bar())
    nuts = bm.GlobalNoUTurnSampler()
    proposers = nuts.get_proposers(world, 10)
    assert all(isinstance(proposer, BaseProposer) for proposer in proposers)


def test_initialize_world():
    model = SampleModel()
    nuts = bm.GlobalNoUTurnSampler()
    world = nuts._initialize_world([model.bar()], {})
    assert model.foo() in world
    assert model.bar() in world
