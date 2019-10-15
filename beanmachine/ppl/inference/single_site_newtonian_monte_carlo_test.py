# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.single_site_newtonian_monte_carlo import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.model.statistical_model import sample


class SingleSiteNewtonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleLogisticRegressionModel(object):
        @sample
        def theta_0(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def theta_1(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def x(self, i):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def y(self, i):
            y = self.theta_1() * self.x(i) + self.theta_0()
            probs = 1 / (1 + (y * -1).exp())
            return dist.Bernoulli(probs)

    def test_single_site_newtonian_monte_carlo(self):
        model = self.SampleNormalModel()
        nw = SingleSiteNewtonianMonteCarlo()
        foo_key = model.foo()
        bar_key = model.bar()
        nw.queries_ = [model.foo()]
        nw.observations_ = {model.bar(): torch.tensor(0.0)}
        nw._infer(10)

        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        self.assertEqual(foo_key in nw.world_.variables_, True)
        self.assertEqual(bar_key in nw.world_.variables_, True)
        self.assertEqual(foo_key in nw.world_.variables_[bar_key].parent, True)
        self.assertEqual(bar_key in nw.world_.variables_[foo_key].children, True)
