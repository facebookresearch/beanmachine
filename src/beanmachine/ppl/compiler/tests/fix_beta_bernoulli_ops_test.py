# Copyright (c) Facebook, Inc. and its affiliates.
"""Compare original and conjugate prior transformed
   Beta-Bernoulli model with operations on Bernoulli samples"""

import random
import unittest

import beanmachine.ppl as bm
import scipy
import torch
from beanmachine.ppl.examples.conjugate_models.beta_bernoulli import BetaBernoulliModel
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


class BetaBernoulliOpsModel(BetaBernoulliModel):
    @bm.functional
    def sum_y(self):
        sum = 0.0
        for i in range(0, 5):
            sum = sum + self.y(i)
        return sum


class BetaBernoulliWithOpsConjugateTest(unittest.TestCase):
    def test_conjugate_graph(self) -> None:
        self.maxDiff = None
        model = BetaBernoulliOpsModel(2.0, 2.0)
        queries = [model.theta(), model.sum_y()]
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
  N00[label=3.0];
  N01[label=5.0];
  N02[label=Beta];
  N03[label=Sample];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=Sample];
  N07[label=Sample];
  N08[label=Sample];
  N09[label=Query];
  N10[label=Sample];
  N11[label=ToPosReal];
  N12[label=ToPosReal];
  N13[label=ToPosReal];
  N14[label=ToPosReal];
  N15[label=ToPosReal];
  N16[label="+"];
  N17[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N03 -> N09;
  N04 -> N05;
  N04 -> N06;
  N04 -> N07;
  N04 -> N08;
  N04 -> N10;
  N05 -> N11;
  N06 -> N12;
  N07 -> N13;
  N08 -> N14;
  N10 -> N15;
  N11 -> N16;
  N12 -> N16;
  N13 -> N16;
  N14 -> N16;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(expected_bmg.strip(), observed_bmg.strip())

    def test_conjugate_samples(self) -> None:
        """
        KS test to check if samples before and after beta-bernoulli conjugate
        transformation is within a certain bound.
        We initialize the seed to ensure the test is deterministic.
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        model = BetaBernoulliOpsModel(2.0, 2.0)
        queries = [model.theta(), model.sum_y()]
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
