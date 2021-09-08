# Copyright (c) Facebook, Inc. and its affiliates.
"""Compare original and conjugate prior transformed
   Beta-Bernoulli model with a hyperparameter given
   by calling a non-random_variable function."""

import random
import unittest

import beanmachine.ppl as bm
import scipy
import torch
import torch.distributions as dist
from beanmachine.ppl.examples.conjugate_models.beta_bernoulli import BetaBernoulliModel
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


class BetaBernoulliScaleHyperParameters(BetaBernoulliModel):
    def scale_alpha(self):
        factor = 2.0
        for i in range(0, 3):
            factor = factor * i
        return factor

    @bm.random_variable
    def theta(self):
        return dist.Beta(self.alpha_ + self.scale_alpha(), self.beta_ + 2.0)


class BetaBernoulliWithScaledHPConjugateTest(unittest.TestCase):
    def test_beta_bernoulli_conjugate_graph(self) -> None:
        model = BetaBernoulliScaleHyperParameters(0.5, 1.5)
        queries = [model.theta()]
        observations = {
            model.y(0): tensor(0.0),
            model.y(1): tensor(0.0),
            model.y(2): tensor(1.0),
            model.y(3): tensor(0.0),
        }
        num_samples = 1000
        bmg = BMGInference()

        # This is the model after beta-bernoulli conjugate rewrite is done
        skip_optimizations = set()
        observed_bmg = bmg.to_dot(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )
        expected_bmg = """
digraph "graph" {
  N0[label=1.5];
  N1[label=6.5];
  N2[label=Beta];
  N3[label=Sample];
  N4[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
}
"""
        self.assertEqual(expected_bmg.strip(), observed_bmg.strip())

    def test_beta_bernoulli_conjugate(self) -> None:
        """
        KS test to check if samples before and after beta-bernoulli conjugate
        transformation is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        model = BetaBernoulliScaleHyperParameters(2.0, 2.0)
        queries = [model.theta()]
        observations = {
            model.y(0): tensor(0.0),
            model.y(1): tensor(0.0),
            model.y(2): tensor(1.0),
            model.y(3): tensor(0.0),
        }
        num_samples = 1000
        bmg = BMGInference()

        posterior_without_opt = bmg.infer(queries, observations, num_samples)
        theta_samples_without_opt = posterior_without_opt[model.theta()][0]

        skip_optimizations = set()
        posterior_with_opt = bmg.infer(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )
        theta_samples_with_opt = posterior_with_opt[model.theta()][0]

        self.assertEqual(
            type(theta_samples_without_opt),
            type(theta_samples_with_opt),
            "Sample type of original and transformed model should be the same.",
        )

        self.assertEqual(
            len(theta_samples_without_opt),
            len(theta_samples_with_opt),
            "Sample size of original and transformed model should be the same.",
        )

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(
                theta_samples_without_opt, theta_samples_with_opt
            ).pvalue,
            0.05,
        )
