# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare original and conjugate prior transformed
   Beta-Binomial model"""

import random
import unittest

import beanmachine.ppl as bm
import scipy
import torch
from beanmachine.ppl.examples.conjugate_models.beta_binomial import BetaBinomialModel
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Beta


class BetaBinomialTransformedModel(BetaBinomialModel):
    """Closed-form Posterior due to conjugacy"""

    @bm.random_variable
    def theta_transformed(self):
        # Analytical posterior Beta(alpha + sum x_i, beta + sum N - sum x_i)
        return Beta(self.alpha_ + 3.0, self.beta_ + (self.n_ - 3.0))


class BetaBinomialConjugateModelTest(unittest.TestCase):
    def test_beta_binomial_conjugate_graph(self) -> None:
        original_model = BetaBinomialModel(2.0, 2.0, 4.0)
        queries = [original_model.theta()]
        observations = {original_model.x(): tensor(3.0)}

        skip_optimizations = set()
        bmg = BMGInference()
        original_graph = bmg.to_dot(
            queries, observations, skip_optimizations=skip_optimizations
        )

        transformed_model = BetaBinomialTransformedModel(2.0, 2.0, 4.0)
        queries_transformed = [transformed_model.theta_transformed()]
        observations_transformed = {}
        transformed_graph = bmg.to_dot(queries_transformed, observations_transformed)

        self.assertEqual(original_graph, transformed_graph)

    def test_beta_binomial_conjugate(self) -> None:
        """
        KS test to check if theta samples from BetaBinomialModel and
        BetaBinomialTransformedModel is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        original_model = BetaBinomialModel(2.0, 2.0, 4.0)
        queries = [original_model.theta()]
        observations = {original_model.x(): tensor(3.0)}

        num_samples = 1000
        bmg = BMGInference()

        posterior_original_model = bmg.infer(queries, observations, num_samples)
        theta_samples_original = posterior_original_model[original_model.theta()][0]

        transformed_model = BetaBinomialTransformedModel(2.0, 2.0, 4.0)
        queries_transformed = [transformed_model.theta_transformed()]
        observations_transformed = {}
        posterior_transformed_model = bmg.infer(
            queries_transformed, observations_transformed, num_samples
        )
        theta_samples_transformed = posterior_transformed_model[
            transformed_model.theta_transformed()
        ][0]

        self.assertEqual(
            type(theta_samples_original),
            type(theta_samples_transformed),
            "Sample type of original and transformed model should be the same.",
        )

        self.assertEqual(
            len(theta_samples_original),
            len(theta_samples_transformed),
            "Sample size of original and transformed model should be the same.",
        )

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(
                theta_samples_original, theta_samples_transformed
            ).pvalue,
            0.05,
        )
