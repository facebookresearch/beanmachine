# Copyright (c) Facebook, Inc. and its affiliates.
import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.global_inference.simple_world import SimpleWorld


class SampleModel:
    @bm.random_variable
    def foo(self):
        return dist.Uniform(0.0, 1.0)

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo(), 1.0)


def test_basic_operations():
    model = SampleModel()
    observations = {model.bar(): torch.rand(())}
    world = SimpleWorld(observations=observations)
    assert world.observations == observations
    assert world._transformed_values == {}
    assert len(world) == 0

    with world:
        model.bar()  # this whould add bar() and its parent foo() to world

    assert len(world) == 2
    assert model.bar() in world
    assert world.transforms[model.bar()] == dist.identity_transform
    assert world.latent_nodes == {model.foo()}

    transformed_foo = world.get_transformed(model.foo())
    assert world.transforms[model.foo()].inv(transformed_foo) == world.call(model.foo())

    del world[model.foo()]
    assert model.foo() not in world


def test_initialize_from_prior():
    model = SampleModel()
    with SimpleWorld(initialize_from_prior=True):
        val1 = model.bar()
    with SimpleWorld(initialize_from_prior=True):
        val2 = model.bar()
    assert val1 != val2


def test_log_prob():
    model = SampleModel()
    world1 = SimpleWorld(observations={model.foo(): torch.tensor(0.0)})
    world1.call(model.bar())
    log_prob1 = world1.log_prob()

    world2 = world1.copy()
    # set to a value with extremely small probability
    world2[model.bar()] = torch.tensor(100.0)
    log_prob2 = world2.log_prob()

    assert log_prob1 > log_prob2
