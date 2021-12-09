# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare original and conjugate prior transformed model"""

import random
import unittest

import scipy
import torch
from beanmachine.ppl.examples.conjugate_models.normal_normal import NormalNormalModel
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Normal


class NormalNormalConjugacyTest(unittest.TestCase):
    def test_conjugate_graph(self) -> None:
        bmg = BMGInference()

        model = NormalNormalModel(10.0, 2.0, 5.0)
        queries = [model.normal_p()]
        observations = {model.normal(): tensor(15.9)}
        observed_bmg = bmg.to_dot(queries, observations, skip_optimizations=set())
        expected_bmg = """
digraph "graph" {
  N0[label=10.813793182373047];
  N1[label=1.8569534304710584];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""
        self.assertEqual(observed_bmg.strip(), expected_bmg.strip())

    def test_normal_normal_conjugate(self) -> None:
        """
        KS test to check if samples from the original NormalNormalModel and
        transformed model is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        true_mu = 0.5
        true_y = Normal(true_mu, 10.0)
        num_samples = 1000
        bmg = BMGInference()

        model = NormalNormalModel(10.0, 2.0, 5.0)
        queries = [model.normal_p()]
        observations = {
            model.normal(): true_y.sample(),
        }

        skip_optimizations = {"NormalNormalConjugateFixer"}
        original_posterior = bmg.infer(
            queries, observations, num_samples, 1, skip_optimizations=skip_optimizations
        )
        original_samples = original_posterior[model.normal_p()][0]

        transformed_posterior = bmg.infer(
            queries, observations, num_samples, 1, skip_optimizations=set()
        )
        transformed_samples = transformed_posterior[model.normal_p()][0]

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
