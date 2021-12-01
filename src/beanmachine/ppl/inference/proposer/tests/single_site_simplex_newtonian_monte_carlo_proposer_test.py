# Copyright (c) Facebook, Inc. and its affiliates
import unittest

import torch
import torch.distributions as dist
from beanmachine import ppl as bm
from beanmachine.ppl.experimental.global_inference.proposer.nmc import (
    SingleSiteSimplexSpaceNMCProposer,
)
from beanmachine.ppl.experimental.global_inference.single_site_nmc import (
    SingleSiteNewtonianMonteCarlo,
)
from beanmachine.ppl.world import World
from torch import tensor


class SingleSiteSimplexNewtonianMonteCarloProposerTest(unittest.TestCase):
    def test_alpha_for_dirichlet(self):
        alpha = tensor([[0.5, 0.5], [0.5, 0.5]])

        @bm.random_variable
        def a():
            return dist.Dirichlet(alpha)

        world_ = World()
        with world_:
            a()
        nw_proposer = SingleSiteSimplexSpaceNMCProposer(a())
        is_valid, predicted_alpha = nw_proposer.compute_alpha(world_)
        self.assertEqual(is_valid, True)
        self.assertAlmostEqual(
            alpha.sum().item(), (predicted_alpha).sum().item(), delta=0.0001
        )

    def test_coin_flip(self):
        prior_heads, prior_tails = 2.0, 2.0
        p = bm.random_variable(lambda: dist.Beta(2.0, 2.0))
        x = bm.random_variable(lambda: dist.Bernoulli(p()))

        heads_observed = 5
        samples = (
            SingleSiteNewtonianMonteCarlo()
            .infer(
                queries=[p()],
                observations={x(): torch.ones(heads_observed)},
                num_samples=100,
                num_chains=1,
            )
            .get_chain(0)
        )

        # assert we are close to the conjugate poserior mean
        self.assertAlmostEqual(
            samples[p()].mean(),
            (prior_heads + heads_observed)
            / (prior_heads + prior_tails + heads_observed),
            delta=0.05,
        )
