# Copyright (c) Facebook, Inc. and its affiliates.
"""Compare original and conjugate prior transformed model"""

import math
import random
import unittest

import beanmachine.ppl as bm
import scipy
import torch
from beanmachine.ppl.examples.conjugate_models.normal_normal import NormalNormalModel
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Normal


class TransformedModel(NormalNormalModel):
    """Conjugate Prior Transformed model"""

    @bm.random_variable
    def normal_p_transformed(self):
        # Analytical posterior computed using transform_mu and transform_sigma_p
        return Normal(self.mu_, self.std_)

    def transform_mu(self, observations):
        precision_prior = pow(self.std_, -2.0)
        precision_data = len(observations) * pow(self.sigma_, -2.0)
        precision_inv = pow((precision_prior + precision_data), -1.0)
        data_sum = sum(observations.values())
        self.mu_ = precision_inv * (
            (self.mu_ * pow(self.std_, -2.0)) + (data_sum * pow(self.sigma_, -2.0))
        )

    def transform_std(self, observations):
        precision_prior = pow(self.std_, -2.0)
        precision_data = len(observations) * pow(self.sigma_, -2.0)
        precision_inv = pow((precision_prior + precision_data), -1.0)
        self.std = math.sqrt(precision_inv)


class NormalNormalConjugacyTest(unittest.TestCase):
    def test_normal_normal_conjugate(self) -> None:
        """
        KS test to check if samples from NormalNormalModel and
        TransformedModel is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        true_mu = 0.5
        true_y = Normal(true_mu, 10.0)
        num_samples = 1000
        bmg = BMGInference()

        original_model = NormalNormalModel(10.0, 2.0, 5.0)
        queries = [original_model.normal_p()]
        observations = {
            original_model.normal(): true_y.sample(),
        }

        original_posterior = bmg.infer(queries, observations, num_samples)
        original_samples = original_posterior[original_model.normal_p()][0]

        transformed_model = TransformedModel(10.0, 2.0, 5.0)
        transformed_model.transform_mu(observations)
        transformed_model.transform_std(observations)
        transformed_queries = [transformed_model.normal_p_transformed()]
        transformed_posterior = bmg.infer(transformed_queries, {}, num_samples)
        transformed_samples = transformed_posterior[
            transformed_model.normal_p_transformed()
        ][0]

        self.assertEqual(
            type(original_samples),
            type(transformed_samples),
            "Sample type of original and transformed model should be the same.",
        )

        self.assertEqual(
            len(original_samples),
            len(transformed_samples),
            "Sample size of original and transformed model should be the same.",
        )

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(original_samples, transformed_samples).pvalue,
            0.05,
        )
