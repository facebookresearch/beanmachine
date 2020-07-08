# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.utils.bm_to_bmg import to_bmg, to_cpp, to_python


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


# These are cases where we just have either a straightforward sample from
# a distribution parameterized with constants, or a distribution parameterized
# with a sample from another distribution.
#
# * No arithmetic
# * No interesting type conversions
# * No use of a sample as an index.
#
source_1 = """
import beanmachine.ppl as bm
import torch
from torch import tensor
from torch.distributions import Bernoulli, Beta, Binomial, HalfCauchy, Normal, StudentT

@bm.random_variable
def flip_straight_constant():
  return Bernoulli(tensor(0.5))

@bm.random_variable
def flip_logit_constant():
  return Bernoulli(logits=tensor(-2.0))

@bm.random_variable
def standard_normal():
  return Normal(0.0, 1.0)

@bm.random_variable
def flip_logit_normal():
  return Bernoulli(logits=standard_normal())

@bm.random_variable
def beta_constant():
  return Beta(1.0, 1.0)

@bm.random_variable
def hc(i):
  return HalfCauchy(1.0)

@bm.random_variable
def beta_hc():
  return Beta(hc(1), hc(2))

@bm.random_variable
def student_t():
  return StudentT(hc(1), standard_normal(), hc(2))

@bm.random_variable
def bin_constant():
  return Binomial(3, 0.5)

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
uint n21 = g.add_constant(3);
uint n22 = g.add_distribution(
  graph::DistributionType::BINOMIAL,
  graph::AtomicType::NATURAL,
  std::vector<uint>({n21, n0}));
uint n23 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n22}));
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 22 ] probability value 0.5
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
Node 21 type 1 parents [ ] children [ 22 ] natural value 3
Node 22 type 2 parents [ 21 0 ] children [ 23 ] unknown value
Node 23 type 3 parents [ 22 ] children [ ] natural value 0
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
n21 = g.add_constant(3)
n22 = g.add_distribution(
  graph.DistributionType.BINOMIAL,
  graph.AtomicType.NATURAL,
  [n21, n0])
n23 = g.add_operator(graph.OperatorType.SAMPLE, [n22])
"""

# These are cases where we have a type conversion on a sample.

source_2 = """
import beanmachine.ppl as bm
import torch
from torch import tensor
from torch.distributions import Bernoulli, Beta, Binomial, HalfCauchy, Normal, StudentT

@bm.random_variable
def flip():
  # Sample is a Boolean
  return Bernoulli(tensor(0.5))

@bm.random_variable
def normal():
  # Converts Boolean to real, positive real
  return Normal(flip(), flip())

@bm.random_variable
def binomial():
  # Converts Boolean to natural and probability
  return Binomial(flip(), flip())

"""

expected_cpp_2 = """
graph::Graph g;
uint n0 = g.add_constant_probability(0.5);
uint n1 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n2}));
uint n4 = g.add_operator(
  graph::OperatorType::TO_POS_REAL, std::vector<uint>({n2}));
uint n5 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n3, n4}));
uint n6 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n5}));
uint n7 = g.add_constant(1);
uint n8 = g.add_constant(0);
n9 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n2, n7, n8}));
uint n10 = g.add_constant_probability(1.0);
uint n11 = g.add_constant_probability(0.0);
n12 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n2, n10, n11}));
uint n13 = g.add_distribution(
  graph::DistributionType::BINOMIAL,
  graph::AtomicType::NATURAL,
  std::vector<uint>({n9, n12}));
uint n14 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n13}));
"""

expected_bmg_2 = """
Node 0 type 1 parents [ ] children [ 1 ] probability value 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown value
Node 2 type 3 parents [ 1 ] children [ 3 4 9 12 ] boolean value 0
Node 3 type 3 parents [ 2 ] children [ 5 ] real value 0
Node 4 type 3 parents [ 2 ] children [ 5 ] pos real value 0
Node 5 type 2 parents [ 3 4 ] children [ 6 ] unknown value
Node 6 type 3 parents [ 5 ] children [ ] real value 0
Node 7 type 1 parents [ ] children [ 9 ] natural value 1
Node 8 type 1 parents [ ] children [ 9 ] natural value 0
Node 9 type 3 parents [ 2 7 8 ] children [ 13 ] natural value 0
Node 10 type 1 parents [ ] children [ 12 ] probability value 1
Node 11 type 1 parents [ ] children [ 12 ] probability value 1e-10
Node 12 type 3 parents [ 2 10 11 ] children [ 13 ] probability value 0
Node 13 type 2 parents [ 9 12 ] children [ 14 ] unknown value
Node 14 type 3 parents [ 13 ] children [ ] natural value 0
"""

expected_python_2 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_probability(0.5)
n1 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.TO_REAL, [n2])
n4 = g.add_operator(graph.OperatorType.TO_POS_REAL, [n2])
n5 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n3, n4])
n6 = g.add_operator(graph.OperatorType.SAMPLE, [n5])
n7 = g.add_constant(1)
n8 = g.add_constant(0)
n9 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n2, n7, n8])
n10 = g.add_constant_probability(1.0)
n11 = g.add_constant_probability(0.0)
n12 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n2, n10, n11])
n13 = g.add_distribution(
  graph.DistributionType.BINOMIAL,
  graph.AtomicType.NATURAL,
  [n9, n12])
n14 = g.add_operator(graph.OperatorType.SAMPLE, [n13])
"""


class EndToEndTest(unittest.TestCase):
    def test_to_cpp_1(self) -> None:
        """test_to_cpp_1 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_1)
        self.assertEqual(observed.strip(), expected_cpp_1.strip())

    def test_to_bmg_1(self) -> None:
        """test_to_bmg_1 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_1).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_1))

    def test_to_python_1(self) -> None:
        """test_to_python_1 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_1)
        self.assertEqual(observed.strip(), expected_python_1.strip())

    def test_to_cpp_2(self) -> None:
        """test_to_cpp_2 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_2)
        self.assertEqual(observed.strip(), expected_cpp_2.strip())

    def test_to_bmg_2(self) -> None:
        """test_to_bmg_2 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_2).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_2))

    def test_to_python_2(self) -> None:
        """test_to_python_2 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_2)
        self.assertEqual(observed.strip(), expected_python_2.strip())
