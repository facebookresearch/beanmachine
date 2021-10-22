# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for tutorial on GMM with Poisson number of components"""
# This file is a manual replica of the Bento tutorial with the same name
# TODO: The disabled test generates the following error:
# E       TypeError: Distribution 'Poisson' is not supported by Bean Machine Graph.
# This will need to be fixed for OSS readiness task

import logging
import unittest

# Comments after imports suggest alternative comment style (for original tutorial)
import beanmachine.ppl as bm
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(42)

# Model


class GaussianMixtureModel(object):
    def __init__(self, K):
        self.K = k

    @bm.random_variable
    def alpha(self, k):
        return dist.Dirichlet(5 * torch.ones(k))

    @bm.random_variable
    def mu(self, c):
        return dist.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=10.0 * torch.eye(2)
        )

    @bm.random_variable
    def sigma(self, c):
        return dist.Gamma(1, 10)

    @bm.random_variable
    def component(self, i):
        alpha = self.alpha(self.K)
        return dist.Categorical(alpha)

    @bm.random_variable
    def y(self, i):
        c = self.component(i).item()
        return dist.MultivariateNormal(
            loc=self.mu(c), covariance_matrix=self.sigma(c) ** 2 * torch.eye(2) + 1e-3
        )


# Creating sample data

n = 32  # num observations
k = 4  # true number of clusters

gmm = GaussianMixtureModel(K=4)

ground_truth = {
    **{gmm.alpha(k): torch.ones(k) * 1.0 / k,},
    **{gmm.mu(i): tensor(i % 2).float() for i in range(k)},
    **{gmm.sigma(i): tensor(0.1) for i in range(k)},
    **{gmm.component(i): tensor(i % k).float() for i in range(n)},
}

# [Visualization code in tutorial skipped]

# Inference parameters
num_samples = (
    1  ###00 Sample size should not affect (the ability to find) compilation issues.
)

queries = (
    [gmm.alpha(gmm.K)]
    + [gmm.component(j) for j in range(n)]
    + [gmm.mu(i) for i in range(k)]
    + [gmm.sigma(i) for i in range(k)]
)

observations = {
    gmm.y(i): ground_truth[gmm.mu(ground_truth[gmm.component(i)].item())]
    for i in range(n)
}


class tutorialGMMwith2DimensionsAnd4Components(unittest.TestCase):
    def test_tutorial_GMM_with_2_dimensions_and_4_components(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        torch.manual_seed(
            42
        )  # Note: Second time we seed. Could be a good tutorial style

        mh = bm.CompositionalInference()
        mh.infer(
            queries, observations, num_samples=num_samples, num_chains=1,
        )

        self.assertTrue(True, msg="We just want to check this point is reached")

    @unittest.skip("TODO: enable when passing")
    def test_tutorial_GMM_with_2_dimensions_and_4_components_to_dot_cpp_python(
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
