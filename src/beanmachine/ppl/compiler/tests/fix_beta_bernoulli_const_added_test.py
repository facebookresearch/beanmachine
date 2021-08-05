# Copyright (c) Facebook, Inc. and its affiliates.
"""Compare original and conjugate prior transformed
   Beta-Bernoulli model with a hyperparameter given
   by calling a non-random_variable function."""

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

    _alpha = 0.5
    _beta = 1.5

    def scale_alpha(self):
        factor = 2.0
        for i in range(0, 3):
            factor = factor * i
        return factor

    @bm.random_variable
    def theta(self):
        return Beta(self._alpha + self.scale_alpha(), self._beta + 2.0)

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
        skip_optimizations = set()
        posterior = bmg.infer(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )
        bmg_graph = bmg.to_dot(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )
        theta_samples = posterior[self.theta()][0]
        return theta_samples, bmg_graph


class HeadsRateModelTransformed(object):
    """Closed-form Posterior due to conjugacy"""

    _alpha = 0.5
    _beta = 1.5

    def scale_alpha(self):
        factor = 2.0
        for i in range(0, 3):
            factor = factor * i
        return factor

    @bm.random_variable
    def theta(self):
        return Beta(self._alpha + self.scale_alpha(), self._beta + 2.0)

    @bm.random_variable
    def y(self, i):
        return Bernoulli(self.theta())

    @bm.random_variable
    def theta_transformed(self):
        # Analytical posterior Beta(alpha + sum y_i, beta + n - sum y_i)
        return Beta(
            self._alpha + self.scale_alpha() + 1.0, self._beta + 2.0 + (4.0 - 1.0)
        )

    def run(self):
        queries_transformed = [self.theta_transformed()]
        observations_transformed = {}
        num_samples = 1000
        bmg = BMGInference()
        posterior_transformed = bmg.infer(
            queries_transformed, observations_transformed, num_samples
        )
        bmg_graph = bmg.to_dot(queries_transformed, observations_transformed)
        theta_samples_transformed = posterior_transformed[self.theta_transformed()][0]
        return theta_samples_transformed, bmg_graph


class HeadsRateModelTest(unittest.TestCase):
    def test_beta_bernoulli_conjugate_graph(self) -> None:
        _, heads_rate_model_graph = HeadsRateModel().run()
        _, heads_rate_model_transformed_graph = HeadsRateModelTransformed().run()

        self.assertEqual(heads_rate_model_graph, heads_rate_model_transformed_graph)

    def test_beta_bernoulli_conjugate(self) -> None:
        """
        KS test to check if HeadsRateModel().run() and HeadsRateModelTransformed().run()
        is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        heads_rate_model_samples, _ = HeadsRateModel().run()
        heads_rate_model_transformed_samples, _ = HeadsRateModelTransformed().run()

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
