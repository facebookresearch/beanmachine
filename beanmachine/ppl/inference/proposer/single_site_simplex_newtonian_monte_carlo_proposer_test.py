# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import torch
import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.inference.proposer.single_site_simplex_newtonian_monte_carlo_proposer import (
    SingleSiteSimplexNewtonianMonteCarloProposer,
)
from beanmachine.ppl.model.statistical_model import sample
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World


class SingleSiteSimplexNewtonianMonteCarloProposerTest(unittest.TestCase):
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

    def test_alpha_for_dirichlet(self):
        alpha = tensor([[0.5, 0.5], [0.5, 0.5]])
        distribution = dist.Dirichlet(alpha)
        val = distribution.sample()
        val.requires_grad_(True)
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            parent=set(),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=val,
            jacobian=tensor(0.0),
        )
        world_ = World()
        nw_proposer = SingleSiteSimplexNewtonianMonteCarloProposer()
        is_valid, predicted_alpha = nw_proposer.compute_alpha(node_var, world_)
        self.assertEqual(is_valid, True)
        self.assertAlmostEqual(
            alpha.sum().item(), (predicted_alpha).sum().item(), delta=0.0001
        )
