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


class DiscreteModel:
    @bm.random_variable
    def foo(self):
        return dist.Categorical(torch.ones(3))

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo().float(), torch.tensor(1.0))


def test_basic_operations():
    model = SampleModel()
    observations = {model.bar(): torch.rand(())}
    world = SimpleWorld(observations=observations)
    assert world.observations == observations
    assert world._variables == {}
    assert len(world) == 0

    with world:
        model.bar()  # this will add bar() and its parent foo() to world

    assert len(world) == 2
    assert model.bar() in world
    assert world._variables[model.bar()].transform == dist.identity_transform
    assert world.latent_nodes == {model.foo()}

    # edge connection
    assert model.foo() in world._variables[model.bar()].parents
    assert model.bar() in world._variables[model.foo()].children
    assert len(world._variables[model.bar()].children) == 0
    assert len(world._variables[model.foo()].parents) == 0

    transformed_foo = world.get_transformed(model.foo())
    assert world._variables[model.foo()].transform.inv(transformed_foo) == world.call(
        model.foo()
    )


def test_initialization():
    model = SampleModel()
    with SimpleWorld():
        val1 = model.bar()
    with SimpleWorld():
        val2 = model.bar()
    assert val1 != val2


def test_log_prob():
    model = SampleModel()
    world1 = SimpleWorld(observations={model.foo(): torch.tensor(0.0)})
    world1.call(model.bar())
    log_prob1 = world1.log_prob()

    world2 = world1.copy()
    # set to a value with extremely small probability
    world2.set_transformed(model.bar(), torch.tensor(100.0))
    log_prob2 = world2.log_prob()

    assert log_prob1 > log_prob2


def test_enumerate():
    model = DiscreteModel()
    world = SimpleWorld(observations={model.bar(): torch.tensor(0.0)})
    with world:
        model.bar()
    assert (torch.tensor([0.0, 1.0, 2.0]) == world.enumerate_node(model.foo())).all()
