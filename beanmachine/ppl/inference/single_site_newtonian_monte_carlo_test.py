# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
import torch.tensor as tensor
import beanmachine.ppl as bm


class SingleSiteNewtonianMonteCarloTest(unittest.TestCase):
    class SampleNormalModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleLogisticRegressionModel(object):
        @bm.random_variable
        def theta_0(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def theta_1(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def x(self, i):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def y(self, i):
            y = self.theta_1() * self.x(i) + self.theta_0()
            probs = 1 / (1 + (y * -1).exp())
            return dist.Bernoulli(probs)

    def test_single_site_newtonian_monte_carlo(self):
        model = self.SampleNormalModel()
        nw = bm.SingleSiteNewtonianMonteCarlo()
        foo_key = model.foo()
        bar_key = model.bar()
        nw.queries_ = [model.foo()]
        nw.observations_ = {model.bar(): torch.tensor(0.0)}
        nw._infer(10)

        world_vars = nw.world_.variables_.vars()
        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        self.assertEqual(foo_key in world_vars, True)
        self.assertEqual(bar_key in world_vars, True)
        self.assertEqual(foo_key in world_vars[bar_key].parent, True)
        self.assertEqual(bar_key in world_vars[foo_key].children, True)
