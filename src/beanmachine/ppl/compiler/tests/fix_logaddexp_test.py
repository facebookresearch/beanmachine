# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Beta


@bm.random_variable
def beta(n):
    return Beta(2, 2)


@bm.functional
def logaddexp_1():
    return torch.logaddexp(beta(0), beta(1))


class FixLogAddExpTest(unittest.TestCase):
    def test_logaddexp_1(self) -> None:
        queries = [logaddexp_1()]
        graph_observed = BMGInference().to_dot(queries, {}, after_transform=False)
        graph_expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=LogAddExp];
  N5[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N4;
  N3 -> N4;
  N4 -> N5;
}
"""
        self.assertEqual(graph_observed.strip(), graph_expected.strip())

    def test_logaddexp_2(self) -> None:
        queries = [logaddexp_1()]
        graph_observed = BMGInference().to_dot(queries, {})
        graph_expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=ToReal];
  N5[label=ToReal];
  N6[label=LogSumExp];
  N7[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N2 -> N4;
  N3 -> N5;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
}
"""
        self.assertEqual(graph_observed.strip(), graph_expected.strip())
