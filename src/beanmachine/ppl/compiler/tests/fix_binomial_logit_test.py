# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare original and conjugate prior transformed
   Beta-Binomial model"""

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import log, tensor
from torch.distributions import Binomial, Normal


@bm.random_variable
def binomial(x):
    return Binomial(100, logits=log(tensor([0.25])))  # equivalent probability: 0.2


@bm.random_variable
def normal(x):
    return Normal(0.0, 1.0)


@bm.random_variable
def binomial_normal_logit():
    return Binomial(100, logits=tensor([normal(0)]))


@bm.functional
def add():
    return binomial(0) + binomial(1)


class BinomialLogitTest(unittest.TestCase):
    def test_constant_binomial_logit_graph(self) -> None:
        observations = {}
        queries_observed = [add()]

        graph_observed = BMGInference().to_dot(queries_observed, observations)

        graph_expected = """
digraph "graph" {
  N0[label=100];
  N1[label=0.20000000298023224];
  N2[label=Binomial];
  N3[label=Sample];
  N4[label=Sample];
  N5[label=ToPosReal];
  N6[label=ToPosReal];
  N7[label="+"];
  N8[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N2 -> N4;
  N3 -> N5;
  N4 -> N6;
  N5 -> N7;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(graph_observed.strip(), graph_expected.strip())

    def test_binomial_normal_logit_graph(self) -> None:
        observations = {}
        queries_observed = [binomial_normal_logit()]

        graph_observed = BMGInference().to_dot(queries_observed, observations)

        graph_expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=100];
  N5[label=Logistic];
  N6[label=Binomial];
  N7[label=Sample];
  N8[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
}
"""
        self.assertEqual(graph_observed.strip(), graph_expected.strip())
