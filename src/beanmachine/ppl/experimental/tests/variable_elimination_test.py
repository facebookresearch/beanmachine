# Copyright (c) Facebook, Inc. and its affiliates.
import os

import beanmachine.ppl as bm
import pytest
import torch
import torch.distributions as dist

if os.environ.get("SANDCASTLE") is None:
    pytest.skip("neuralpp unavailable outside of Facebook", allow_module_level=True)
else:
    from beanmachine.ppl.experimental.variable_elimination.variable_elimination import (
        make_neuralpp_factor,
    )
    from beanmachine.ppl.world import World
    from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
        PyTorchTableFactor,
    )
    from neuralpp.inference.graphical_model.variable.integer_variable import (
        IntegerVariable,
    )


class DiscreteModel:
    @bm.random_variable
    def foo(self):
        return dist.Categorical(torch.ones(3))

    @bm.random_variable
    def foo2(self):
        idx = torch.rand(3)[self.foo().long()]
        return dist.Categorical(torch.tensor([idx, 1 - idx]))

    @bm.random_variable
    def bar(self):
        return dist.Normal(self.foo2().float(), torch.tensor(1.0))


def test_prior_probs():
    # exponentially enumerate every node in the graph and retrieve its logpmf
    model = DiscreteModel()
    world = World(observations={model.bar(): torch.tensor(0.0)})
    sups = []
    with world:
        model.bar()
        for v in world.latent_nodes:
            dist = v.function(*v.arguments)
            if dist.has_enumerate_support:
                # 0 indexed support dists
                for i in range(len(dist.probs)):
                    sups.append((v.function, i, dist.probs[i]))
    print(sups)


def test_bm_neuralpp_factor_conversion():
    data = [
        (
            # input
            # Factor on A, B with domain sizes 2, 3 and respective probabilities
            (("A", "B"), {"A": 2, "B": 3}, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            # output
            # neuralpp factor on same variables, probability tensor, indicating log space storage
            PyTorchTableFactor(
                (
                    IntegerVariable("A", 2),
                    IntegerVariable("B", 3),
                ),
                torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(2, 3),
                log_space=True,
            ),
        ),
        (
            # input
            # Factor on A with domain sizes 2 and respective probabilities
            (("A",), {"A": 2}, [0.1, 0.9]),
            # output
            # neuralpp factor on same variables, probability tensor, indicating log space storage
            PyTorchTableFactor(
                (IntegerVariable("A", 2),),
                torch.tensor([0.1, 0.9]),
                log_space=True,
            ),
        ),
    ]
    for input, output in data:
        assert output == make_neuralpp_factor(*input)
