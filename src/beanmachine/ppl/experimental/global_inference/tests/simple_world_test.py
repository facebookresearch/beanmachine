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


class DynamicModel:
    @bm.random_variable
    def foo(self):
        return dist.Bernoulli(0.5)

    @bm.random_variable
    def bar(self, i: int):
        return dist.Normal(0.0, 1.0)

    @bm.random_variable
    def baz(self):
        mu = self.bar(int(self.foo()))
        return dist.Normal(mu, 1.0)


def test_basic_operations():
    model = SampleModel()
    observations = {model.bar(): torch.rand(())}
    world = SimpleWorld(observations=observations)
    assert world.observations == observations
    assert len(world.latent_nodes) == 0
    assert len(world) == 0

    with world:
        model.bar()  # this will add bar() and its parent foo() to world

    assert len(world) == 2
    assert model.bar() in world
    assert world.get_variable(model.bar()).transform == dist.identity_transform
    assert world.latent_nodes == {model.foo()}

    # edge connection
    assert model.foo() in world.get_variable(model.bar()).parents
    assert model.bar() in world.get_variable(model.foo()).children
    assert len(world.get_variable(model.bar()).children) == 0
    assert len(world.get_variable(model.foo()).parents) == 0

    transformed_foo = world.get_transformed(model.foo())
    assert world.get_variable(model.foo()).transform.inv(transformed_foo) == world.call(
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

    # set to a value with extremely small probability
    world2 = world1.replace_transformed({model.bar(): torch.tensor(100.0)})
    log_prob2 = world2.log_prob()

    assert log_prob1 > log_prob2


def test_enumerate():
    model = DiscreteModel()
    world = SimpleWorld(observations={model.bar(): torch.tensor(0.0)})
    with world:
        model.bar()
    assert (torch.tensor([0.0, 1.0, 2.0]) == world.enumerate_node(model.foo())).all()


def test_change_parents():
    model = DynamicModel()
    world = SimpleWorld(initialize_fn=lambda d: torch.zeros_like(d.sample()))
    with world:
        model.baz()

    assert model.foo() in world.get_variable(model.baz()).parents
    assert model.bar(0) in world.get_variable(model.baz()).parents
    assert model.bar(1) not in world.get_variable(model.baz()).parents
    assert model.baz() in world.get_variable(model.bar(0)).children

    world2 = world.replace_transformed({model.foo(): torch.tensor(1.0)})

    assert model.bar(0) not in world2.get_variable(model.baz()).parents
    assert model.bar(1) in world2.get_variable(model.baz()).parents
    assert model.baz() in world2.get_variable(model.bar(1)).children
    assert model.baz() not in world2.get_variable(model.bar(0)).children
