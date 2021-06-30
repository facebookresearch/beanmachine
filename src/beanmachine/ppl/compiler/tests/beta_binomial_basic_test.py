# Copyright (c) Facebook, Inc. and its affiliates.
"""Compare original and conjugate prior transformed model"""

import random
import unittest

import beanmachine.ppl as bm
import scipy
import torch
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


class HeadsRateModel(object):
    """Original, untransformed model"""

    @bm.random_variable
    def theta(self):
        return Beta(2.0, 2.0)

    @bm.random_variable
    def y(self, i):
        return Bernoulli(self.theta())

    def run(self):
        queries = [self.theta()]
        observations = {
            self.y(0): tensor(0.0),
            self.y(1): tensor(0.0),
            self.y(2): tensor(1.0),
            self.y(3): tensor(0.0),
        }
        num_samples = 1000
        bmg = BMGInference()
        posterior = bmg.infer(queries, observations, num_samples)
        theta_samples = posterior[self.theta()][0]
        return theta_samples


class HeadsRateModelTransformed(object):
    """Conjugate Prior Transformed model"""

    @bm.random_variable
    def theta(self):
        return Beta(2.0, 2.0)

    @bm.random_variable
    def y(self, i):
        return Bernoulli(self.theta())

    @bm.random_variable
    def theta_transformed(self):
        # Analytical posterior Beta(alpha + sum y_i, beta + n - sum y_i)
        return Beta(2.0 + 1.0, 2.0 + (4.0 - 1.0))

    def run(self):
        # queries = [self.theta()]
        queries_transformed = [self.theta_transformed()]
        # observations = {
        #     self.y(0): tensor(0.0),
        #     self.y(1): tensor(0.0),
        #     self.y(2): tensor(1.0),
        #     self.y(3): tensor(0.0),
        # }
        observations_transformed = {}
        num_samples = 1000
        bmg = BMGInference()
        # posterior = bmg.infer(queries, observations, num_samples)
        posterior_transformed = bmg.infer(
            queries_transformed, observations_transformed, num_samples
        )
        # theta_samples = posterior[self.theta](0)
        theta_samples_transformed = posterior_transformed[self.theta_transformed()][0]
        return theta_samples_transformed


class HeadsRateModelTest(unittest.TestCase):
    def test_beta_binomial_conjugate(self) -> None:
        """
        KS test to check if HeadsRateModel().run() and HeadsRateModelTransformed().run()
        is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        heads_rate_model_samples = HeadsRateModel().run()
        heads_rate_model_transformed_samples = HeadsRateModelTransformed().run()

        self.assertEqual(
            type(heads_rate_model_samples),
            type(heads_rate_model_transformed_samples),
            "Sample type of original and transformed model should be the same.",
        )

        self.assertEqual(
            len(heads_rate_model_samples),
            len(heads_rate_model_transformed_samples),
            "Sample size of original and transformed model should be the same.",
        )

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(
                heads_rate_model_samples, heads_rate_model_transformed_samples
            ).pvalue,
            0.05,
        )
