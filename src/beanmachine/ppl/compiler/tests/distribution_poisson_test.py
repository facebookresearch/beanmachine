# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for poisson distribution"""


import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Gamma, Poisson


@bm.random_variable
def poisson_1():
    return Poisson(rate=0.5)


@bm.random_variable
def gamma_1():
    return Gamma(1.0, 4.0)


@bm.random_variable
def poisson_2():
    return Poisson(rate=gamma_1())


@bm.random_variable
def poisson_3():
    return Poisson(rate=-1 * gamma_1())


@bm.random_variable
def poisson_4():
    return Poisson(rate=tensor([1.0, 2.0]))


class distributionPoissonTest(unittest.TestCase):
    def test_graphs_poisson_with_constant_rate(self) -> None:
        self.maxDiff = None

        queries = [poisson_1()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Poisson];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed_cpp = BMGInference().to_cpp(queries, observations)
        expected_cpp = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(0.5);
uint n1 = g.add_distribution(
  graph::DistributionType::POISSON,
  graph::AtomicType::NATURAL,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint q0 = g.query(n2);
"""
        self.assertEqual(expected_cpp.strip(), observed_cpp.strip())

        observed_python = BMGInference().to_python(queries, observations)
        expected_python = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(0.5)
n1 = g.add_distribution(
  graph.DistributionType.POISSON,
  graph.AtomicType.NATURAL,
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
q0 = g.query(n2)
"""
        self.assertEqual(expected_python.strip(), observed_python.strip())

    def test_poisson_rate_with_sample_from_distribution(self) -> None:
        self.maxDiff = None

        queries = [poisson_2()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=4.0];
  N2[label=Gamma];
  N3[label=Sample];
  N4[label=Poisson];
  N5[label=Sample];
  N6[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_poisson_tensor_input(self) -> None:
        self.maxDiff = None

        queries = [poisson_4()]
        observations = {}
        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=Poisson];
  N2[label=Sample];
  N3[label=2.0];
  N4[label=Poisson];
  N5[label=Sample];
  N6[label=2];
  N7[label=1];
  N8[label=ToMatrix];
  N9[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N8;
  N3 -> N4;
  N4 -> N5;
  N5 -> N8;
  N6 -> N8;
  N7 -> N8;
  N8 -> N9;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_poisson_rate_error_reporting(self) -> None:
        self.maxDiff = None

        queries = [poisson_3()]
        observations = {}
        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot(queries, observations)
        self.assertEqual(
            str(ex.exception),
            "The rate of a Poisson is required to be a positive real but is a negative real.\n"
            "The Poisson was created in function call poisson_3().",
            msg="Poisson distribution with non-positive real rates should throw an exception.",
        )
