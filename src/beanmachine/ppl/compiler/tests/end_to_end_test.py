# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.compiler.bm_to_bmg import to_bmg, to_cpp, to_python


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
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Chi2,
    Gamma,
    HalfCauchy,
    Normal,
    StudentT,
    Uniform,
)

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

@bm.random_variable
def gamma():
  return Gamma(1.0, 2.0)

@bm.random_variable
def flat():
  return Uniform(0.0, 1.0)

@bm.random_variable
def chi2():
  return Chi2(8.0)
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
uint n24 = g.add_constant_pos_real(2.0);
uint n25 = g.add_distribution(
  graph::DistributionType::GAMMA,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n7, n24}));
uint n26 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n25}));
uint n27 = g.add_distribution(
  graph::DistributionType::FLAT,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({}));
uint n28 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n27}));
uint n29 = g.add_constant_pos_real(4.0);
uint n30 = g.add_constant_pos_real(0.5);
uint n31 = g.add_distribution(
  graph::DistributionType::GAMMA,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n29, n30}));
uint n32 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n31}));
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 22 ] probability 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ ] boolean 0
Node 3 type 1 parents [ ] children [ 4 ] probability 0.119203
Node 4 type 2 parents [ 3 ] children [ 5 ] unknown
Node 5 type 3 parents [ 4 ] children [ ] boolean 0
Node 6 type 1 parents [ ] children [ 8 ] real 0
Node 7 type 1 parents [ ] children [ 8 12 12 14 25 ] positive real 1
Node 8 type 2 parents [ 6 7 ] children [ 9 ] unknown
Node 9 type 3 parents [ 8 ] children [ 10 19 ] real 0
Node 10 type 2 parents [ 9 ] children [ 11 ] unknown
Node 11 type 3 parents [ 10 ] children [ ] boolean 0
Node 12 type 2 parents [ 7 7 ] children [ 13 ] unknown
Node 13 type 3 parents [ 12 ] children [ ] probability 1e-10
Node 14 type 2 parents [ 7 ] children [ 15 16 ] unknown
Node 15 type 3 parents [ 14 ] children [ 17 19 ] positive real 1e-10
Node 16 type 3 parents [ 14 ] children [ 17 19 ] positive real 1e-10
Node 17 type 2 parents [ 15 16 ] children [ 18 ] unknown
Node 18 type 3 parents [ 17 ] children [ ] probability 1e-10
Node 19 type 2 parents [ 15 9 16 ] children [ 20 ] unknown
Node 20 type 3 parents [ 19 ] children [ ] real 0
Node 21 type 1 parents [ ] children [ 22 ] natural 3
Node 22 type 2 parents [ 21 0 ] children [ 23 ] unknown
Node 23 type 3 parents [ 22 ] children [ ] natural 0
Node 24 type 1 parents [ ] children [ 25 ] positive real 2
Node 25 type 2 parents [ 7 24 ] children [ 26 ] unknown
Node 26 type 3 parents [ 25 ] children [ ] positive real 1e-10
Node 27 type 2 parents [ ] children [ 28 ] unknown
Node 28 type 3 parents [ 27 ] children [ ] probability 1e-10
Node 29 type 1 parents [ ] children [ 31 ] positive real 4
Node 30 type 1 parents [ ] children [ 31 ] positive real 0.5
Node 31 type 2 parents [ 29 30 ] children [ 32 ] unknown
Node 32 type 3 parents [ 31 ] children [ ] positive real 1e-10
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
n24 = g.add_constant_pos_real(2.0)
n25 = g.add_distribution(
  graph.DistributionType.GAMMA,
  graph.AtomicType.POS_REAL,
  [n7, n24])
n26 = g.add_operator(graph.OperatorType.SAMPLE, [n25])
n27 = g.add_distribution(
  graph.DistributionType.FLAT,
  graph.AtomicType.PROBABILITY,
  [])
n28 = g.add_operator(graph.OperatorType.SAMPLE, [n27])
n29 = g.add_constant_pos_real(4.0)
n30 = g.add_constant_pos_real(0.5)
n31 = g.add_distribution(
  graph.DistributionType.GAMMA,
  graph.AtomicType.POS_REAL,
  [n29, n30])
n32 = g.add_operator(graph.OperatorType.SAMPLE, [n31])
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
Node 0 type 1 parents [ ] children [ 1 ] probability 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 4 9 12 ] boolean 0
Node 3 type 3 parents [ 2 ] children [ 5 ] real 0
Node 4 type 3 parents [ 2 ] children [ 5 ] positive real 1e-10
Node 5 type 2 parents [ 3 4 ] children [ 6 ] unknown
Node 6 type 3 parents [ 5 ] children [ ] real 0
Node 7 type 1 parents [ ] children [ 9 ] natural 1
Node 8 type 1 parents [ ] children [ 9 ] natural 0
Node 9 type 3 parents [ 2 7 8 ] children [ 13 ] natural 0
Node 10 type 1 parents [ ] children [ 12 ] probability 1
Node 11 type 1 parents [ ] children [ 12 ] probability 1e-10
Node 12 type 3 parents [ 2 10 11 ] children [ 13 ] probability 1e-10
Node 13 type 2 parents [ 9 12 ] children [ 14 ] unknown
Node 14 type 3 parents [ 13 ] children [ ] natural 0
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

# Here we multiply a bool by a natural, and then use that as a natural.
# This cannot be turned into a BMG that uses multiplication because
# there is no multiplication defined on naturals or bools; the best
# we could do as a multiplication is to turn both into a positive real
# and multiply those.  But we *can* turn this into an if-then-else
# that takes a bool and returns either the given natural or zero,
# so that's what we'll do.

source_3 = """
import beanmachine.ppl as bm
import torch
from torch import tensor
from torch.distributions import Bernoulli, Binomial

@bm.random_variable
def flip():
  return Bernoulli(0.5)

@bm.random_variable
def nat():
  return Binomial(2, 0.5)

@bm.random_variable
def bin():
  return Binomial(nat() * flip(), 0.5)
"""

expected_python_3 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_probability(0.5)
n1 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_constant(2)
n4 = g.add_distribution(
  graph.DistributionType.BINOMIAL,
  graph.AtomicType.NATURAL,
  [n3, n0])
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
n6 = g.add_constant(0)
n7 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n2, n5, n6])
n8 = g.add_distribution(
  graph.DistributionType.BINOMIAL,
  graph.AtomicType.NATURAL,
  [n7, n0])
n9 = g.add_operator(graph.OperatorType.SAMPLE, [n8])
"""

# End-to-end tests for math functions

source_4 = """
import beanmachine.ppl as bm
import torch
from torch import tensor
from torch.distributions import HalfCauchy, Normal, Bernoulli

@bm.random_variable
def pos(n):
  return HalfCauchy(1.0)

@bm.random_variable
def math():
  return Normal(pos(0).log(), pos(1).exp())

@bm.random_variable
def math2():
  return HalfCauchy(pos(2) ** pos(3))

@bm.random_variable
def math3():
  # PHI
  return Bernoulli(Normal(0.0, 1.0).cdf(pos(4)))
"""

expected_python_4 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(1.0)
n1 = g.add_distribution(
  graph.DistributionType.HALF_CAUCHY,
  graph.AtomicType.POS_REAL,
  [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n4 = g.add_operator(graph.OperatorType.LOG, [n2])
n5 = g.add_operator(graph.OperatorType.EXP, [n3])
n6 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n4, n5])
n7 = g.add_operator(graph.OperatorType.SAMPLE, [n6])
n8 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n9 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n10 = g.add_operator(graph.OperatorType.POW, [n8, n9])
n11 = g.add_distribution(
  graph.DistributionType.HALF_CAUCHY,
  graph.AtomicType.POS_REAL,
  [n10])
n12 = g.add_operator(graph.OperatorType.SAMPLE, [n11])
n13 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n14 = g.add_operator(graph.OperatorType.TO_REAL, [n13])
n15 = g.add_operator(graph.OperatorType.PHI, [n14])
n16 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n15])
n17 = g.add_operator(graph.OperatorType.SAMPLE, [n16])
"""

expected_cpp_4 = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(1.0);
uint n1 = g.add_distribution(
  graph::DistributionType::HALF_CAUCHY,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n4 = g.add_operator(
  graph::OperatorType::LOG, std::vector<uint>({n2}));
uint n5 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n3}));
uint n6 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n4, n5}));
uint n7 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n9 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n10 = g.add_operator(
  graph::OperatorType::POW, std::vector<uint>({n8, n9}));
uint n11 = g.add_distribution(
  graph::DistributionType::HALF_CAUCHY,
  graph::AtomicType::POS_REAL,
  std::vector<uint>({n10}));
uint n12 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n11}));
uint n13 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n14 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n13}));
uint n15 = g.add_operator(
  graph::OperatorType::PHI, std::vector<uint>({n14}));
uint n16 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n15}));
uint n17 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n16}));
"""

expected_bmg_4 = """
Node 0 type 1 parents [ ] children [ 1 ] positive real 1
Node 1 type 2 parents [ 0 ] children [ 2 3 8 9 13 ] unknown
Node 2 type 3 parents [ 1 ] children [ 4 ] positive real 1e-10
Node 3 type 3 parents [ 1 ] children [ 5 ] positive real 1e-10
Node 4 type 3 parents [ 2 ] children [ 6 ] real 0
Node 5 type 3 parents [ 3 ] children [ 6 ] positive real 1e-10
Node 6 type 2 parents [ 4 5 ] children [ 7 ] unknown
Node 7 type 3 parents [ 6 ] children [ ] real 0
Node 8 type 3 parents [ 1 ] children [ 10 ] positive real 1e-10
Node 9 type 3 parents [ 1 ] children [ 10 ] positive real 1e-10
Node 10 type 3 parents [ 8 9 ] children [ 11 ] positive real 1e-10
Node 11 type 2 parents [ 10 ] children [ 12 ] unknown
Node 12 type 3 parents [ 11 ] children [ ] positive real 1e-10
Node 13 type 3 parents [ 1 ] children [ 14 ] positive real 1e-10
Node 14 type 3 parents [ 13 ] children [ 15 ] real 0
Node 15 type 3 parents [ 14 ] children [ 16 ] probability 1e-10
Node 16 type 2 parents [ 15 ] children [ 17 ] unknown
Node 17 type 3 parents [ 16 ] children [ ] boolean 0
"""

# Demonstrate that we generate 1-p as a complement

source_5 = """
import beanmachine.ppl as bm
import torch
from torch.distributions import Bernoulli, Beta

@bm.random_variable
def beta():
  return Beta(2.0, 2.0)

@bm.random_variable
def flip():
  return Bernoulli(1.0 - beta())

"""

expected_python_5 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(2.0)
n1 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n0, n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.COMPLEMENT, [n2])
n4 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n3])
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
"""

expected_cpp_5 = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(2.0);
uint n1 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n0, n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::COMPLEMENT, std::vector<uint>({n2}));
uint n4 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n3}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
"""

expected_bmg_5 = """
Node 0 type 1 parents [ ] children [ 1 1 ] positive real 2
Node 1 type 2 parents [ 0 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 ] probability 1e-10
Node 3 type 3 parents [ 2 ] children [ 4 ] probability 1e-10
Node 4 type 2 parents [ 3 ] children [ 5 ] unknown
Node 5 type 3 parents [ 4 ] children [ ] boolean 0
"""


# Demonstrate that we generate -log(prob) as a positive real.

source_6 = """
import beanmachine.ppl as bm
import torch
from torch.distributions import Beta

@bm.random_variable
def beta1():
  return Beta(2.0, 2.0)

@bm.random_variable
def beta2():
  return Beta(-beta1().log(), 2.0)
"""

expected_python_6 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(2.0)
n1 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n0, n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.NEGATIVE_LOG, [n2])
n4 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n3, n0])
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
"""

expected_cpp_6 = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(2.0);
uint n1 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n0, n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::NEGATIVE_LOG, std::vector<uint>({n2}));
uint n4 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n3, n0}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
"""

expected_bmg_6 = """
Node 0 type 1 parents [ ] children [ 1 1 4 ] positive real 2
Node 1 type 2 parents [ 0 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 ] probability 1e-10
Node 3 type 3 parents [ 2 ] children [ 4 ] positive real 1e-10
Node 4 type 2 parents [ 3 0 ] children [ 5 ] unknown
Node 5 type 3 parents [ 4 ] children [ ] probability 1e-10
"""

# Demonstrate that identity additions and multiplications
# are removed from the graph.  Here we are computing
# 0 + 0 * hc(0) + 1 * hc(1) + 2 * hc(2)
# but as you can see, in the final program we generate
# the code as though we had written hc(1) + 2 * hc(2).

source_7 = """
import beanmachine.ppl as bm
import torch
from torch.distributions import Beta, HalfCauchy

@bm.random_variable
def hc(n):
  return HalfCauchy(3.0)

@bm.random_variable
def beta2():
  s = 0.0
  for i in [0, 1, 2]:
    s = s + i * hc(i)
  return Beta(s, 4.0)
"""

expected_python_7 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_pos_real(3.0)
n1 = g.add_distribution(
  graph.DistributionType.HALF_CAUCHY,
  graph.AtomicType.POS_REAL,
  [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n4 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n5 = g.add_constant_pos_real(2.0)
n6 = g.add_operator(graph.OperatorType.MULTIPLY, [n5, n4])
n7 = g.add_operator(graph.OperatorType.ADD, [n3, n6])
n8 = g.add_constant_pos_real(4.0)
n9 = g.add_distribution(
  graph.DistributionType.BETA,
  graph.AtomicType.PROBABILITY,
  [n7, n8])
n10 = g.add_operator(graph.OperatorType.SAMPLE, [n9])
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

    def test_to_python_3(self) -> None:
        """test_to_python_3 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_3)
        self.assertEqual(observed.strip(), expected_python_3.strip())

    def test_to_python_4(self) -> None:
        """test_to_python_4 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_4)
        self.assertEqual(observed.strip(), expected_python_4.strip())

    def test_to_cpp_4(self) -> None:
        """test_to_cpp_4 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_4)
        self.assertEqual(observed.strip(), expected_cpp_4.strip())

    def test_to_bmg_4(self) -> None:
        """test_to_bmg_4 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_4).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_4))

    def test_to_python_5(self) -> None:
        """test_to_python_5 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_5)
        self.assertEqual(observed.strip(), expected_python_5.strip())

    def test_to_cpp_5(self) -> None:
        """test_to_cpp_5 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_5)
        self.assertEqual(observed.strip(), expected_cpp_5.strip())

    def test_to_bmg_5(self) -> None:
        """test_to_bmg_5 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_5).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_5))

    def test_to_python_6(self) -> None:
        """test_to_python_6 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_6)
        self.assertEqual(observed.strip(), expected_python_6.strip())

    def test_to_cpp_6(self) -> None:
        """test_to_cpp_6 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_cpp(source_6)
        self.assertEqual(observed.strip(), expected_cpp_6.strip())

    def test_to_bmg_6(self) -> None:
        """test_to_bmg_6 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_bmg(source_6).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_6))

    def test_to_python_7(self) -> None:
        """test_to_python_7 from end_to_end_test.py"""
        self.maxDiff = None
        observed = to_python(source_7)
        self.assertEqual(observed.strip(), expected_python_7.strip())
