# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import HalfCauchy, Normal


@bm.random_variable
def hc(n):
    return HalfCauchy(1)


@bm.random_variable
def normal():
    return Normal(0, 1)


@bm.functional
def logprob():
    # Demonstrate that we can apply operators other than
    # sample to stochastic distributions.
    normal_sample = normal()  # Sample
    normal_dist_1 = Normal(0, hc(1))
    normal_dist_2 = Normal(0, hc(2))
    # "instance receiver" form
    weight_1 = normal_dist_1.log_prob(normal_sample)
    # "static receiver" form
    weight_2 = Normal.log_prob(normal_dist_2, normal_sample)
    # Non-stochastic distribution, stochastic value
    weight_3 = Normal(2, 3).log_prob(normal_sample)
    return weight_1 + weight_2 + weight_3


class LogProbTest(unittest.TestCase):
    def test_logprob(self) -> None:
        self.maxDiff = None
        queries = [logprob()]
        observed = BMGInference().to_dot(queries, {})
        expected = """
digraph "graph" {
  N00[label=1.0];
  N01[label=HalfCauchy];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=Normal];
  N05[label=Sample];
  N06[label=Sample];
  N07[label=Normal];
  N08[label=LogProb];
  N09[label=Normal];
  N10[label=LogProb];
  N11[label=2.0];
  N12[label=3.0];
  N13[label=Normal];
  N14[label=LogProb];
  N15[label="+"];
  N16[label=Query];
  N00 -> N01;
  N00 -> N04;
  N01 -> N02;
  N01 -> N06;
  N02 -> N07;
  N03 -> N04;
  N03 -> N07;
  N03 -> N09;
  N04 -> N05;
  N05 -> N08;
  N05 -> N10;
  N05 -> N14;
  N06 -> N09;
  N07 -> N08;
  N08 -> N15;
  N09 -> N10;
  N10 -> N15;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
}
      """
        self.assertEqual(expected.strip(), observed.strip())
