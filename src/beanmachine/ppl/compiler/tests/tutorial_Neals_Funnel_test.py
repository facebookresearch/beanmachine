# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for tutorial on Sparse Logistic Regression"""
# This file is a manual replica of the Bento tutorial with the same name
# TODO: Running the disabled test produces the following error:
# E           ValueError: Fetching the value of attribute log_prob is not supported in Bean Machine Graph.
# This is a block for Beanstalk OSS readiness
# TODO: Check imports for conistency

import logging
import unittest

import beanmachine.ppl as bm
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.inference.bmg_inference import BMGInference
from beanmachine.ppl.inference.proposer.single_site_hamiltonian_monte_carlo_proposer import (
    SingleSiteHamiltonianMonteCarloProposer,
)
from torch import tensor

# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(11)

# Model

xs, zs = torch.meshgrid(
    torch.arange(-50, 50, 0.1),
    torch.arange(-15.0, 15.0, 0.1),
)
density = (
    dist.Normal(0.0, (zs / 2.0).exp()).log_prob(xs).exp()
    * dist.Normal(0.0, 3.0).log_prob(zs).exp()
)


@bm.random_variable
def z():
    """
    An uninformative (flat) prior for z.
    """
    # TODO(tingley): Replace with Flat once it's part of the framework.
    return dist.Normal(0, 10000)


@bm.random_variable
def x():
    """
    An uninformative (flat) prior for x.
    """
    # TODO(tingley): Replace with Flat once it's part of the framework.
    return dist.Normal(0, 10000)


@bm.random_variable
def neals_funnel_coin_flip():
    """
    Flip a "coin", which is heads with probability equal to the probability
    of drawing z and x from the true Neal's funnel posterior.
    """
    return dist.Bernoulli(
        (
            dist.Normal(0.0, (z() / 2.0).exp()).log_prob(x())
            + dist.Normal(0.0, 3.0).log_prob(z())
        ).exp()
    )


# Inference parameters

num_samples = 1  ###000
num_chains = 4

observations = {neals_funnel_coin_flip(): tensor(1.0)}

queries = [z(), x()]


class tutorialNealsFunnelTest(unittest.TestCase):
    def test_tutorial_Neals_Funnel(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        # Note: No explicit seed here (in original tutorial model). Should we add one?
        nmc = bm.SingleSiteNewtonianMonteCarlo()
        samples_nmc = nmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        hmc = bm.SingleSiteHamiltonianMonteCarlo(path_length=0.1, step_size=0.01)
        samples_hmc = hmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        ghmc = bm.CompositionalInference(
            {
                z: SingleSiteHamiltonianMonteCarloProposer(
                    path_length=0.1, step_size=0.01
                ),
                x: SingleSiteHamiltonianMonteCarloProposer(
                    path_length=0.1, step_size=0.01
                ),
            }
        )
        ghmc.add_sequential_proposer([z, x])
        samples_ghmc = ghmc.infer(
            queries=queries,
            observations=observations,
            num_samples=num_samples,
            num_chains=num_chains,
        )

        self.assertTrue(True, msg="We just want to check this point is reached")

    def disabled_test_tutorial_Neals_Funnel_to_dot_cpp_python(
        self,
    ) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())
