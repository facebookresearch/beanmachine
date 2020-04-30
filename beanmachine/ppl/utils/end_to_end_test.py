# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.utils.bm_to_bmg import to_bmg, to_cpp, to_python


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


source_1 = """
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import tensor
from torch.distributions import Bernoulli, Beta, HalfCauchy, Normal, StudentT

@sample
def flip_straight_constant():
  return Bernoulli(tensor(0.5))

@sample
def flip_logit_constant():
  return Bernoulli(logits=tensor(-2.0))

@sample
def standard_normal():
  return Normal(0.0, 1.0)

@sample
def flip_logit_normal():
  return Bernoulli(logits=standard_normal())

@sample
def beta_constant():
  return Beta(1.0, 1.0)

@sample
def hc(i):
  return HalfCauchy(1.0)

@sample
def beta_hc():
  return Beta(hc(1), hc(2))

@sample
def student_t():
  return StudentT(hc(1), standard_normal(), hc(2))

"""

expected_cpp_1 = """
graph::Graph g;
uint n0 = g.add_constant_probability(0.5);
uint n1 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_constant_probability(0.11920291930437088);
uint n4 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n3}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
uint n6 = g.add_constant(0.0);
uint n7 = g.add_constant_pos_real(1.0);
uint n8 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n6, n7}));
uint n9 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n8}));
uint n10 = g.add_distribution(
  graph::DistributionType::BERNOULLI_LOGIT,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n9}));
uint n11 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n10}));
uint n12 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n7, n7}));
uint n13 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n12}));
uint n14 = g.add_distribution(
  graph::DistributionType::HALF_CAUCHY,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n7}));
uint n15 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n14}));
uint n16 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n14}));
uint n17 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n15, n16}));
uint n18 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n17}));
uint n19 = g.add_distribution(
  graph::DistributionType::STUDENT_T,
  graph::AtomicType::REAL,
  std::vector<uint>({n15, n9, n16}));
uint n20 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n19}));
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 ] probability value 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown value
Node 2 type 3 parents [ 1 ] children [ ] boolean value 0
Node 3 type 1 parents [ ] children [ 4 ] probability value 0.119203
Node 4 type 2 parents [ 3 ] children [ 5 ] unknown value
Node 5 type 3 parents [ 4 ] children [ ] boolean value 0
Node 6 type 1 parents [ ] children [ 8 ] real value 0
Node 7 type 1 parents [ ] children [ 8 12 12 14 ] pos real value 1
Node 8 type 2 parents [ 6 7 ] children [ 9 ] unknown value
Node 9 type 3 parents [ 8 ] children [ 10 19 ] real value 0
Node 10 type 2 parents [ 9 ] children [ 11 ] unknown value
Node 11 type 3 parents [ 10 ] children [ ] boolean value 0
Node 12 type 2 parents [ 7 7 ] children [ 13 ] unknown value
Node 13 type 3 parents [ 12 ] children [ ] probability value 0
Node 14 type 2 parents [ 7 ] children [ 15 16 ] unknown value
Node 15 type 3 parents [ 14 ] children [ 17 19 ] pos real value 0
Node 16 type 3 parents [ 14 ] children [ 17 19 ] pos real value 0
Node 17 type 2 parents [ 15 16 ] children [ 18 ] unknown value
Node 18 type 3 parents [ 17 ] children [ ] probability value 0
Node 19 type 2 parents [ 15 9 16 ] children [ 20 ] unknown value
Node 20 type 3 parents [ 19 ] children [ ] real value 0
"""

expected_python_1 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_probability(0.5)
n1 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_constant_probability(0.11920291930437088)
n4 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n3])
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
n6 = g.add_constant(0.0)
n7 = g.add_constant_pos_real(1.0)
n8 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n6, n7])
n9 = g.add_operator(graph.OperatorType.SAMPLE, [n8])
n10 = g.add_distribution(
  graph.DistributionType.BERNOULLI_LOGIT,
  graph.AtomicType.BOOLEAN,
  [n9])
n11 = g.add_operator(graph.OperatorType.SAMPLE, [n10])
n12 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n7, n7])
n13 = g.add_operator(graph.OperatorType.SAMPLE, [n12])
n14 = g.add_distribution(
  graph.DistributionType.HALF_CAUCHY,
  graph.AtomicType.POS_REAL,
  [n7])
n15 = g.add_operator(graph.OperatorType.SAMPLE, [n14])
n16 = g.add_operator(graph.OperatorType.SAMPLE, [n14])
n17 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n15, n16])
n18 = g.add_operator(graph.OperatorType.SAMPLE, [n17])
n19 = g.add_distribution(
  graph.DistributionType.STUDENT_T,
  graph.AtomicType.REAL,
  [n15, n9, n16])
n20 = g.add_operator(graph.OperatorType.SAMPLE, [n19])
"""


class EndToEndTest(unittest.TestCase):
    def test_to_cpp(self) -> None:
        """test_to_cpp from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_1)
        self.assertEqual(observed.strip(), expected_cpp_1.strip())

    def test_to_bmg(self) -> None:
        """test_to_bmg from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_1).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_1))

    def test_to_python(self) -> None:
        """test_to_python from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_1)
        self.assertEqual(observed.strip(), expected_python_1.strip())
