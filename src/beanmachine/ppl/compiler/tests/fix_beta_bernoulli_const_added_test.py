# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare original and conjugate prior transformed
   Beta-Bernoulli model with a hyperparameter given
   by calling a non-random_variable function."""

import unittest

from beanmachine.ppl.compiler.testlib.conjugate_models import (
    BetaBernoulliScaleHyperParameters,
)
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor


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
