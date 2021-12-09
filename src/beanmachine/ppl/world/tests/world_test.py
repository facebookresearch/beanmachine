# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.world import World


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


class ChangeSupportModel:
    @bm.random_variable
    def foo(self):
        return dist.Bernoulli(0.3)

    @bm.random_variable
    def bar(self):
        if self.foo():
            return dist.Categorical(logits=torch.rand((3,)))
        else:
            return dist.Normal(0.0, 1.0)

    @bm.random_variable
    def baz(self):
        return dist.Bernoulli(self.foo())


def test_basic_operations():
    model = SampleModel()
    observations = {model.bar(): torch.rand(())}
    world = World(observations=observations)
    assert world.observations == observations
    assert len(world.latent_nodes) == 0
    assert len(world) == 0

    with world:
        model.bar()  # this will add bar() and its parent foo() to world

    assert len(world) == 2
    assert model.bar() in world
    assert world.latent_nodes == {model.foo()}

    # edge connection
    assert model.foo() in world.get_variable(model.bar()).parents
    assert model.bar() in world.get_variable(model.foo()).children
    assert len(world.get_variable(model.bar()).children) == 0
    assert len(world.get_variable(model.foo()).parents) == 0

    assert world.get_variable(model.foo()).value == world.call(model.foo())


def test_initialization():
    model = SampleModel()
    with World():
        val1 = model.bar()
    with World():
        val2 = model.bar()
    assert val1 != val2


def test_log_prob():
    model = SampleModel()
    world1 = World(observations={model.foo(): torch.tensor(0.0)})
    world1.call(model.bar())
    log_prob1 = world1.log_prob()

    # set to a value with extremely small probability
    world2 = world1.replace({model.bar(): torch.tensor(100.0)})
    log_prob2 = world2.log_prob()

    assert log_prob1 > log_prob2


def test_enumerate():
    model = DiscreteModel()
    world = World(observations={model.bar(): torch.tensor(0.0)})
    with world:
        model.bar()
    assert (torch.tensor([0.0, 1.0, 2.0]) == world.enumerate_node(model.foo())).all()


def test_change_parents():
    model = DynamicModel()
    world = World(initialize_fn=lambda d: torch.zeros_like(d.sample()))
    with world:
        model.baz()

    assert model.foo() in world.get_variable(model.baz()).parents
    assert model.bar(0) in world.get_variable(model.baz()).parents
    assert model.bar(1) not in world.get_variable(model.baz()).parents
    assert model.baz() in world.get_variable(model.bar(0)).children

    world2 = world.replace({model.foo(): torch.tensor(1.0)})

    assert model.bar(0) not in world2.get_variable(model.baz()).parents
    assert model.bar(1) in world2.get_variable(model.baz()).parents
    assert model.baz() in world2.get_variable(model.bar(1)).children
    assert model.baz() not in world2.get_variable(model.bar(0)).children


def test_distribution_and_log_prob_update():
    model = ChangeSupportModel()
    with World(observations={model.baz(): torch.tensor(1.0)}) as world:
        model.bar()
        model.baz()

    world = world.replace({model.foo(): torch.tensor(0.0)})
    world2 = world.replace({model.foo(): torch.tensor(1.0)})

    bar_var = world.get_variable(model.bar())
    assert isinstance(bar_var.distribution, dist.Normal)

    bar_var2 = world2.get_variable(model.bar())
    assert isinstance(bar_var2.distribution, dist.Categorical)

    # verify that the children's log prob is recomputed when foo gets updated
    baz_var = world.get_variable(model.baz())  # Bernoulli(0.0)
    baz_var2 = world2.get_variable(model.baz())  # Bernoulli(1.0)
    # recall that baz() is observed to be 1.0
    assert baz_var.log_prob < baz_var2.log_prob
