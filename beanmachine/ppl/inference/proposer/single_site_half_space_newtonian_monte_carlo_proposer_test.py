# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_half_space_newtonian_monte_carlo_proposer import (
    SingleSiteHalfSpaceNewtonianMonteCarloProposer,
)
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


class SingleSiteHalfSpaceNewtonianMonteCarloProposerTest(unittest.TestCase):
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

    class SampleFallbackModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Gamma(tensor(2.0), tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_alpha_and_beta_for_gamma(self):
        alpha = tensor([2.0, 2.0, 2.0])
        beta = tensor([2.0, 2.0, 2.0])
        distribution = dist.Gamma(alpha, beta)
        val = distribution.sample()
        val.requires_grad_(True)
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            is_discrete=False,
            transforms=[],
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world_ = World()
        nw_proposer = SingleSiteHalfSpaceNewtonianMonteCarloProposer()
        is_valid, predicted_alpha, predicted_beta = nw_proposer.compute_alpha_beta(
            node_var, world_
        )
        self.assertEqual(is_valid, True)
        self.assertAlmostEqual(
            alpha.sum().item(), (predicted_alpha).sum().item(), delta=0.0001
        )
        self.assertAlmostEqual(
            beta.sum().item(), (predicted_beta).sum().item(), delta=0.0001
        )
