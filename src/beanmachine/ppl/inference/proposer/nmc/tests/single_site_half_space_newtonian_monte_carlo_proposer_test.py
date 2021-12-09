# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.nmc import (
    SingleSiteHalfSpaceNMCProposer,
)
from beanmachine.ppl.world import World
from torch import tensor


class SingleSiteHalfSpaceNewtonianMonteCarloProposerTest(unittest.TestCase):
    class SampleNormalModel:
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class SampleLogisticRegressionModel:
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

    class SampleFallbackModel:
        @bm.random_variable
        def foo(self):
            return dist.Gamma(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_alpha_and_beta_for_gamma(self):
        alpha = tensor([2.0, 2.0, 2.0])
        beta = tensor([2.0, 2.0, 2.0])

        @bm.random_variable
        def gamma():
            return dist.Gamma(alpha, beta)

        world = World()
        with world:
            gamma()
        nw_proposer = SingleSiteHalfSpaceNMCProposer(gamma())
        is_valid, predicted_alpha, predicted_beta = nw_proposer.compute_alpha_beta(
            world
        )
        self.assertEqual(is_valid, True)
        self.assertAlmostEqual(
            alpha.sum().item(), (predicted_alpha).sum().item(), delta=0.0001
        )
        self.assertAlmostEqual(
            beta.sum().item(), (predicted_beta).sum().item(), delta=0.0001
        )
