# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_simplex_newtonian_monte_carlo_proposer import (
    SingleSiteSimplexNewtonianMonteCarloProposer,
)
from beanmachine.ppl.world.variable import Variable
from beanmachine.ppl.world.world import World
from torch import tensor


class SingleSiteSimplexNewtonianMonteCarloProposerTest(unittest.TestCase):
    def test_alpha_for_dirichlet(self):
        alpha = tensor([[0.5, 0.5], [0.5, 0.5]])
        distribution = dist.Dirichlet(alpha)
        val = distribution.sample()
        val.requires_grad_(True)
        node_var = Variable(
            distribution=distribution,
            value=val,
            log_prob=distribution.log_prob(val).sum(),
            transformed_value=val,
            jacobian=tensor(0.0),
        )
        world_ = World()
        nw_proposer = SingleSiteSimplexNewtonianMonteCarloProposer()
        is_valid, predicted_alpha = nw_proposer.compute_alpha(node_var, world_)
        self.assertEqual(is_valid, True)
        self.assertAlmostEqual(
            alpha.sum().item(), (predicted_alpha).sum().item(), delta=0.0001
        )
