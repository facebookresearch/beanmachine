# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch import exp, log, logsumexp
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def log_1():
    return log(exp(norm(0)) + exp(norm(1)) + exp(norm(2)))


@bm.functional
def logsumexp_1():
    return logsumexp(torch.tensor([norm(0), norm(1), norm(2)]), 0)


@bm.functional
def log_2():
    return log_1()


@bm.functional
def log_3():
    return logsumexp_1()


@bm.functional
def exp_1():
    return exp(norm(3)) + exp(norm(4))


@bm.functional
def log_4():
    return log(exp(norm(0)) + exp(norm(1)) + exp(norm(2)) + exp_1())


@bm.functional
def log_5():
    return log_4()


@bm.functional
def log_6():
    return log_4() + exp_1()


class FixLogSumExpTest(unittest.TestCase):
    def test_fix_log_sum_exp_1(self) -> None:
        observations = {}
        queries_observed = [log_2()]

        graph_observed = BMGInference().to_dot(queries_observed, observations)

        queries_expected = [log_3()]

        graph_expected = BMGInference().to_dot(queries_expected, observations)

        self.assertEqual(graph_observed.strip(), graph_expected.strip())

    def test_fix_log_sum_exp_2(self) -> None:
        observations = {}
        queries_observed = [log_5()]

        graph_observed = BMGInference().to_dot(queries_observed, observations)

        graph_expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Sample];
  N5[label=Sample];
  N6[label=Sample];
  N7[label=Sample];
  N8[label=LogSumExp];
  N9[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N2 -> N4;
  N2 -> N5;
  N2 -> N6;
  N2 -> N7;
  N3 -> N8;
  N4 -> N8;
  N5 -> N8;
  N6 -> N8;
  N7 -> N8;
  N8 -> N9;
}
"""
        self.assertEqual(graph_observed.strip(), graph_expected.strip())

    def test_fix_log_sum_exp_3(self) -> None:
        observations = {}
        queries_observed = [log_6()]

        graph_observed = BMGInference().to_dot(queries_observed, observations)

        graph_expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label=Sample];
  N07[label=Sample];
  N08[label=Exp];
  N09[label=Exp];
  N10[label=Exp];
  N11[label=Exp];
  N12[label=Exp];
  N13[label="+"];
  N14[label="+"];
  N15[label=Log];
  N16[label=ToReal];
  N17[label="+"];
  N18[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N02 -> N06;
  N02 -> N07;
  N03 -> N08;
  N04 -> N09;
  N05 -> N10;
  N06 -> N11;
  N07 -> N12;
  N08 -> N14;
  N09 -> N14;
  N10 -> N14;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N13 -> N16;
  N14 -> N15;
  N15 -> N17;
  N16 -> N17;
  N17 -> N18;
}
"""

        self.assertEqual(graph_observed.strip(), graph_expected.strip())
