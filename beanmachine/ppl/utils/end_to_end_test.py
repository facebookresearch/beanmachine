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
from torch.distributions import Bernoulli, Normal

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
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 ] probability value 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown value
Node 2 type 3 parents [ 1 ] children [ ] boolean value 0
Node 3 type 1 parents [ ] children [ 4 ] probability value 0.119203
Node 4 type 2 parents [ 3 ] children [ 5 ] unknown value
Node 5 type 3 parents [ 4 ] children [ ] boolean value 0
Node 6 type 1 parents [ ] children [ 8 ] real value 0
Node 7 type 1 parents [ ] children [ 8 ] pos real value 1
Node 8 type 2 parents [ 6 7 ] children [ 9 ] unknown value
Node 9 type 3 parents [ 8 ] children [ 10 ] real value 0
Node 10 type 2 parents [ 9 ] children [ 11 ] unknown value
Node 11 type 3 parents [ 10 ] children [ ] boolean value 0
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
