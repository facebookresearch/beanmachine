# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test of realistic linear regression model"""
import unittest

from beanmachine.ppl.compiler.bm_to_bmg import to_bmg, to_cpp, to_dot, to_python


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


source = """
import beanmachine.ppl as bm
import torch
from torch.distributions import Normal, Uniform

@bm.random_variable
def theta_0():
    return Normal(0,1)

@bm.random_variable
def theta_1():
    return Normal(0,1)

@bm.random_variable
def error():
    return Uniform(0,1)

@bm.random_variable
def x(i):
    return Normal(0,1)

@bm.random_variable
def y(i):
    return Normal(theta_0() + theta_1() * x(i), error())

y(0)
y(1)
"""

expected_cpp = """
graph::Graph g;
uint n0 = g.add_constant(0.0);
uint n1 = g.add_constant_pos_real(1.0);
uint n2 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n0, n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n6 = g.add_distribution(
  graph::DistributionType::FLAT,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({}));
uint n7 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n4, n5}));
uint n9 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n3, n8}));
uint n10 = g.add_operator(
  graph::OperatorType::TO_POS_REAL, std::vector<uint>({n7}));
uint n11 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n9, n10}));
uint n12 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n11}));
uint n13 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n14 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n4, n13}));
uint n15 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n3, n14}));
uint n16 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n15, n10}));
uint n17 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n16}));
"""

expected_bmg = """
Node 0 type 1 parents [ ] children [ 2 ] real 0
Node 1 type 1 parents [ ] children [ 2 ] positive real 1
Node 2 type 2 parents [ 0 1 ] children [ 3 4 5 13 ] unknown
Node 3 type 3 parents [ 2 ] children [ 9 15 ] real 0
Node 4 type 3 parents [ 2 ] children [ 8 14 ] real 0
Node 5 type 3 parents [ 2 ] children [ 8 ] real 0
Node 6 type 2 parents [ ] children [ 7 ] unknown
Node 7 type 3 parents [ 6 ] children [ 10 ] probability 1e-10
Node 8 type 3 parents [ 4 5 ] children [ 9 ] real 0
Node 9 type 3 parents [ 3 8 ] children [ 11 ] real 0
Node 10 type 3 parents [ 7 ] children [ 11 16 ] positive real 1e-10
Node 11 type 2 parents [ 9 10 ] children [ 12 ] unknown
Node 12 type 3 parents [ 11 ] children [ ] real 0
Node 13 type 3 parents [ 2 ] children [ 14 ] real 0
Node 14 type 3 parents [ 4 13 ] children [ 15 ] real 0
Node 15 type 3 parents [ 3 14 ] children [ 16 ] real 0
Node 16 type 2 parents [ 15 10 ] children [ 17 ] unknown
Node 17 type 3 parents [ 16 ] children [ ] real 0
"""

expected_python = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(0.0)
n1 = g.add_constant_pos_real(1.0)
n2 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n0, n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n6 = g.add_distribution(
  graph.DistributionType.FLAT,
  graph.AtomicType.PROBABILITY,
  [])
n7 = g.add_operator(graph.OperatorType.SAMPLE, [n6])
n8 = g.add_operator(graph.OperatorType.MULTIPLY, [n4, n5])
n9 = g.add_operator(graph.OperatorType.ADD, [n3, n8])
n10 = g.add_operator(graph.OperatorType.TO_POS_REAL, [n7])
n11 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n9, n10])
n12 = g.add_operator(graph.OperatorType.SAMPLE, [n11])
n13 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n14 = g.add_operator(graph.OperatorType.MULTIPLY, [n4, n13])
n15 = g.add_operator(graph.OperatorType.ADD, [n3, n14])
n16 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n15, n10])
n17 = g.add_operator(graph.OperatorType.SAMPLE, [n16])
"""

expected_dot = """
digraph "graph" {
  N00[label="0.0:R"];
  N01[label="1.0:R+"];
  N02[label="Normal:R"];
  N03[label="Sample:R"];
  N04[label="Sample:R"];
  N05[label="Sample:R"];
  N06[label="Flat:P"];
  N07[label="Sample:P"];
  N08[label="*:R"];
  N09[label="+:R"];
  N10[label="ToPosReal:R+"];
  N11[label="Normal:R"];
  N12[label="Sample:R"];
  N13[label="Sample:R"];
  N14[label="*:R"];
  N15[label="+:R"];
  N16[label="Normal:R"];
  N17[label="Sample:R"];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N04[label=operand];
  N02 -> N05[label=operand];
  N02 -> N13[label=operand];
  N03 -> N09[label=left];
  N03 -> N15[label=left];
  N04 -> N08[label=left];
  N04 -> N14[label=left];
  N05 -> N08[label=right];
  N06 -> N07[label=operand];
  N07 -> N10[label=operand];
  N08 -> N09[label=right];
  N09 -> N11[label=mu];
  N10 -> N11[label=sigma];
  N10 -> N16[label=sigma];
  N11 -> N12[label=operand];
  N13 -> N14[label=right];
  N14 -> N15[label=right];
  N15 -> N16[label=mu];
  N16 -> N17[label=operand];
}
"""


class LinearRegressionTest(unittest.TestCase):
    def test_to_cpp(self) -> None:
        """test_to_cpp from linear_regression_test.py"""
        self.maxDiff = None
        observed = to_cpp(source)
        self.assertEqual(observed.strip(), expected_cpp.strip())

    def test_to_bmg(self) -> None:
        """test_to_bmg from linear_regression_test.py"""
        self.maxDiff = None
        observed = to_bmg(source).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg))

    def test_to_python(self) -> None:
        """test_to_python from linear_regression_test.py"""
        self.maxDiff = None
        observed = to_python(source)
        self.assertEqual(observed.strip(), expected_python.strip())

    def test_to_dot(self) -> None:
        """test_to_dot from linear_regression_test.py"""
        self.maxDiff = None
        observed = to_dot(
            source=source,
            graph_types=True,
            inf_types=False,
            edge_requirements=False,
            point_at_input=True,
            after_transform=True,
        )
        self.assertEqual(observed.strip(), expected_dot.strip())
