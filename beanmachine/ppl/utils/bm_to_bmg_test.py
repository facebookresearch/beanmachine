# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.utils.bm_to_bmg import (
    to_bmg,
    to_cpp,
    to_dot,
    to_python,
    to_python_raw,
)
from beanmachine.ppl.utils.memoize import RecursionError


# flake8 does not provide any mechanism to disable warnings in
# multi-line strings, so just turn it off for this file.
# flake8: noqa


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


source1 = """
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli

assert 1 + 1 == 2, "math still works"

@sample
def X():
  assert 2 + 2 == 4, "math still works"
  return Bernoulli(tensor(0.01))

@sample
def Y():
  return Bernoulli(tensor(0.01))

@sample
def Z():
  return Bernoulli(
    1 - exp(log(tensor(0.99)) + X() * log(tensor(0.01)) + Y() * log(tensor(0.01)))
  )
"""

expected_raw_1 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli


@probabilistic(bmg)
@memoize
def X():
    a7 = 0.01
    a4 = bmg.handle_function(tensor, [a7])
    r1 = bmg.handle_function(Bernoulli, [a4])
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def Y():
    a8 = 0.01
    a5 = bmg.handle_function(tensor, [a8])
    r2 = bmg.handle_function(Bernoulli, [a5])
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def Z():
    a9 = 1
    a16 = bmg.handle_dot_get(torch, 'tensor')
    a19 = -0.010050326585769653
    a14 = bmg.handle_function(a16, [a19])
    a20 = bmg.handle_function(X, [])
    a24 = bmg.handle_dot_get(torch, 'tensor')
    a26 = -4.605170249938965
    a22 = bmg.handle_function(a24, [a26])
    a17 = bmg.handle_multiplication(a20, a22)
    a13 = bmg.handle_addition(a14, a17)
    a18 = bmg.handle_function(Y, [])
    a23 = bmg.handle_dot_get(torch, 'tensor')
    a25 = -4.605170249938965
    a21 = bmg.handle_function(a23, [a25])
    a15 = bmg.handle_multiplication(a18, a21)
    a12 = bmg.handle_addition(a13, a15)
    a11 = bmg.handle_function(exp, [a12])
    a10 = bmg.handle_negate(a11)
    a6 = bmg.handle_addition(a9, a10)
    r3 = bmg.handle_function(Bernoulli, [a6])
    return bmg.handle_sample(r3)


roots = [X(), Y(), Z()]
bmg.remove_orphans(roots)
"""

expected_python_1 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(tensor(0.009999999776482582))
n1 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n4 = g.add_constant(1.0)
n5 = g.add_constant(tensor(-0.010050326585769653))
n6 = g.add_constant(tensor(-4.605170249938965))
n7 = g.add_operator(graph.OperatorType.MULTIPLY, [n2, n6])
n8 = g.add_operator(graph.OperatorType.ADD, [n5, n7])
n9 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n6])
n10 = g.add_operator(graph.OperatorType.ADD, [n8, n9])
n11 = g.add_operator(graph.OperatorType.EXP, [n10])
n12 = g.add_operator(graph.OperatorType.NEGATE, [n11])
n13 = g.add_operator(graph.OperatorType.ADD, [n4, n12])
n14 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n13])
n15 = g.add_operator(graph.OperatorType.SAMPLE, [n14])
"""

expected_dot_1 = """
digraph "graph" {
  N0[label=0.009999999776482582];
  N10[label="+"];
  N11[label=Exp];
  N12[label="-"];
  N13[label="+"];
  N14[label=Bernoulli];
  N15[label=Sample];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=1];
  N5[label=-0.010050326585769653];
  N6[label=-4.605170249938965];
  N7[label="*"];
  N8[label="+"];
  N9[label="*"];
  N1 -> N0[label=probability];
  N10 -> N8[label=left];
  N10 -> N9[label=right];
  N11 -> N10[label=operand];
  N12 -> N11[label=operand];
  N13 -> N12[label=right];
  N13 -> N4[label=left];
  N14 -> N13[label=probability];
  N15 -> N14[label=operand];
  N2 -> N1[label=operand];
  N3 -> N1[label=operand];
  N7 -> N2[label=left];
  N7 -> N6[label=right];
  N8 -> N5[label=left];
  N8 -> N7[label=right];
  N9 -> N3[label=left];
  N9 -> N6[label=right];
}
"""

expected_cpp_1 = """
graph::Graph g;
uint n0 = g.add_constant(0.009999999776482582);
uint n1 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n4 = g.add_constant(1.0);
uint n5 = g.add_constant(torch::from_blob((float[]){-0.010050326585769653}, {}));
uint n6 = g.add_constant(torch::from_blob((float[]){-4.605170249938965}, {}));
uint n7 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n2, n6}));
uint n8 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n5, n7}));
uint n9 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n3, n6}));
uint n10 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n8, n9}));
uint n11 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n10}));
uint n12 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n11}));
uint n13 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n4, n12}));
uint n14 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n13}));
uint n15 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n14}));
uint n16 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n15}));
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 ] real value 0.01
Node 1 type 2 parents [ 0 ] children [ 2 3 ] unknown value
Node 2 type 3 parents [ 1 ] children [ 7 ] unknown value
Node 3 type 3 parents [ 1 ] children [ 9 ] unknown value
Node 4 type 1 parents [ ] children [ 13 ] real value 1
Node 5 type 1 parents [ ] children [ 8 ] tensor value -0.0100503
[ CPUFloatType{} ]
Node 6 type 1 parents [ ] children [ 7 9 ] tensor value -4.60517
[ CPUFloatType{} ]
Node 7 type 3 parents [ 2 6 ] children [ 8 ] unknown value
Node 8 type 3 parents [ 5 7 ] children [ 10 ] unknown value
Node 9 type 3 parents [ 3 6 ] children [ 10 ] unknown value
Node 10 type 3 parents [ 8 9 ] children [ 11 ] unknown value
Node 11 type 3 parents [ 10 ] children [ 12 ] unknown value
Node 12 type 3 parents [ 11 ] children [ 13 ] unknown value
Node 13 type 3 parents [ 4 12 ] children [ 14 ] unknown value
Node 14 type 3 parents [ 13 ] children [ 15 ] unknown value
Node 15 type 2 parents [ 14 ] children [ 16 ] unknown value
Node 16 type 3 parents [ 15 ] children [ ] unknown value
"""


source2 = """
import torch
from torch import exp, log, tensor, neg
from torch.distributions import Bernoulli

@sample
def x(n):
  return Bernoulli(tensor(0.5) + log(exp(n * tensor(0.1))))

@sample
def z():
  return Bernoulli(tensor(0.3) ** x(0) + x(0) / tensor(10.0) - neg(x(1) * tensor(0.4)))
"""

# In the medium term, we need to create a mechanism in BMG to represent
# "random variable with index".  This in particular will be necessary
# for scenarios like "Bernoulli(y(x())"; suppose x is a random variable
# either True and False and y(n) is a random variable that takes in True
# or False and produces a sample from 0.0 to 1.0. We do not have
# a way in BMG today to represent this because we require exactly as many
# sample nodes in the graph as there are samples in the program.
#
# However, because we do hoist the indices of x(0) and x(1) as nodes
# in the graph here, and because nodes are deduplicated, we end
# up doing the right thing when the indices are constants.

expected_raw_2 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
import torch
from torch import exp, log, tensor, neg
from torch.distributions import Bernoulli


@probabilistic(bmg)
@memoize
def x(n):
    a7 = 0.5
    a5 = bmg.handle_function(tensor, [a7])
    a25 = 0.1
    a20 = bmg.handle_function(tensor, [a25])
    a15 = bmg.handle_multiplication(n, a20)
    a11 = bmg.handle_function(exp, [a15])
    a8 = bmg.handle_function(log, [a11])
    a3 = bmg.handle_addition(a5, a8)
    r1 = bmg.handle_function(Bernoulli, [a3])
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def z():
    a16 = 0.3
    a12 = bmg.handle_function(tensor, [a16])
    a21 = 0
    a17 = bmg.handle_function(x, [a21])
    a9 = bmg.handle_power(a12, a17)
    a22 = 0
    a18 = bmg.handle_function(x, [a22])
    a26 = 10.0
    a23 = bmg.handle_function(tensor, [a26])
    a13 = bmg.handle_division(a18, a23)
    a6 = bmg.handle_addition(a9, a13)
    a27 = 1
    a24 = bmg.handle_function(x, [a27])
    a29 = 0.4
    a28 = bmg.handle_function(tensor, [a29])
    a19 = bmg.handle_multiplication(a24, a28)
    a14 = bmg.handle_function(neg, [a19])
    a10 = bmg.handle_negate(a14)
    a4 = bmg.handle_addition(a6, a10)
    r2 = bmg.handle_function(Bernoulli, [a4])
    return bmg.handle_sample(r2)


roots = [z()]
bmg.remove_orphans(roots)
"""

expected_dot_2 = """
digraph "graph" {
  N0[label=0.30000001192092896];
  N10[label=Sample];
  N11[label=0.4000000059604645];
  N12[label="*"];
  N13[label="-"];
  N14[label="-"];
  N15[label="+"];
  N16[label=Bernoulli];
  N17[label=Sample];
  N1[label=0.5];
  N2[label=Bernoulli];
  N3[label=Sample];
  N4[label="**"];
  N5[label=10.0];
  N6[label="/"];
  N7[label="+"];
  N8[label=0.6000000238418579];
  N9[label=Bernoulli];
  N10 -> N9[label=operand];
  N12 -> N10[label=left];
  N12 -> N11[label=right];
  N13 -> N12[label=operand];
  N14 -> N13[label=operand];
  N15 -> N14[label=right];
  N15 -> N7[label=left];
  N16 -> N15[label=probability];
  N17 -> N16[label=operand];
  N2 -> N1[label=probability];
  N3 -> N2[label=operand];
  N4 -> N0[label=left];
  N4 -> N3[label=right];
  N6 -> N3[label=left];
  N6 -> N5[label=right];
  N7 -> N4[label=left];
  N7 -> N6[label=right];
  N9 -> N8[label=probability];
}
"""

# As mentioned in the comment above, we will need a way to represent indexed
# samples, but until we have that, I've implemented a simple loop unroller
# that we can use to experiment with these sorts of distributions:

source3 = """
import torch
from torch import exp, log, tensor, neg
from torch.distributions import Bernoulli

@sample
def x(n):
  return Bernoulli(tensor(0.5) + log(exp(n * tensor(0.1))))

@sample
def z():
  sum = 0.0
  for n in [0, 1]:
      sum = sum + log(tensor(0.01)) * x(n)
  return Bernoulli(
    1 - exp(log(tensor(0.99)) + sum)
  )
"""

expected_raw_3 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
import torch
from torch import exp, log, tensor, neg
from torch.distributions import Bernoulli


@probabilistic(bmg)
@memoize
def x(n):
    a15 = 0.5
    a9 = bmg.handle_function(tensor, [a15])
    a26 = 0.1
    a24 = bmg.handle_function(tensor, [a26])
    a22 = bmg.handle_multiplication(n, a24)
    a20 = bmg.handle_function(exp, [a22])
    a16 = bmg.handle_function(log, [a20])
    a5 = bmg.handle_addition(a9, a16)
    r1 = bmg.handle_function(Bernoulli, [a5])
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def z():
    sum = 0.0
    n = 0
    a10 = bmg.handle_dot_get(torch, 'tensor')
    a17 = -4.605170249938965
    a6 = bmg.handle_function(a10, [a17])
    a11 = bmg.handle_function(x, [n])
    a2 = bmg.handle_multiplication(a6, a11)
    sum = bmg.handle_addition(sum, a2)
    n = 1
    a12 = bmg.handle_dot_get(torch, 'tensor')
    a18 = -4.605170249938965
    a7 = bmg.handle_function(a12, [a18])
    a13 = bmg.handle_function(x, [n])
    a3 = bmg.handle_multiplication(a7, a13)
    sum = bmg.handle_addition(sum, a3)
    a14 = 1
    a27 = bmg.handle_dot_get(torch, 'tensor')
    a28 = -0.010050326585769653
    a25 = bmg.handle_function(a27, [a28])
    a23 = bmg.handle_addition(a25, sum)
    a21 = bmg.handle_function(exp, [a23])
    a19 = bmg.handle_negate(a21)
    a8 = bmg.handle_addition(a14, a19)
    r4 = bmg.handle_function(Bernoulli, [a8])
    return bmg.handle_sample(r4)


roots = [z()]
bmg.remove_orphans(roots)
"""

expected_dot_3 = """
digraph "graph" {
  N0[label=1];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label="*"];
  N13[label="+"];
  N14[label="+"];
  N15[label=Exp];
  N16[label="-"];
  N17[label="+"];
  N18[label=Bernoulli];
  N19[label=Sample];
  N1[label=-0.010050326585769653];
  N2[label=0.0];
  N3[label=-4.605170249938965];
  N4[label=0.5];
  N5[label=Bernoulli];
  N6[label=Sample];
  N7[label="*"];
  N8[label="+"];
  N9[label=0.6000000238418579];
  N10 -> N9[label=probability];
  N11 -> N10[label=operand];
  N12 -> N11[label=right];
  N12 -> N3[label=left];
  N13 -> N12[label=right];
  N13 -> N8[label=left];
  N14 -> N13[label=right];
  N14 -> N1[label=left];
  N15 -> N14[label=operand];
  N16 -> N15[label=operand];
  N17 -> N0[label=left];
  N17 -> N16[label=right];
  N18 -> N17[label=probability];
  N19 -> N18[label=operand];
  N5 -> N4[label=probability];
  N6 -> N5[label=operand];
  N7 -> N3[label=left];
  N7 -> N6[label=right];
  N8 -> N2[label=left];
  N8 -> N7[label=right];
}
"""

source4 = """
import torch
from torch import exp, log, tensor, neg
from torch.distributions import Bernoulli

@sample
def x(n):
  return Bernoulli(tensor(0.5) + log(exp(n * tensor(0.1))))

@sample
def z():
  sum = 0.0
  a = 2
  b = 3
  for n in [a, a+b]:
      sum = sum + log(tensor(0.01)) * x(n)
  return Bernoulli(
    1 - exp(log(tensor(0.99)) + sum)
  )
"""

expected_raw_4 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
import torch
from torch import exp, log, tensor, neg
from torch.distributions import Bernoulli


@probabilistic(bmg)
@memoize
def x(n):
    a11 = 0.5
    a8 = bmg.handle_function(tensor, [a11])
    a23 = 0.1
    a21 = bmg.handle_function(tensor, [a23])
    a19 = bmg.handle_multiplication(n, a21)
    a16 = bmg.handle_function(exp, [a19])
    a12 = bmg.handle_function(log, [a16])
    a4 = bmg.handle_addition(a8, a12)
    r1 = bmg.handle_function(Bernoulli, [a4])
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def z():
    sum = 0.0
    a = 2
    b = 3
    a5 = bmg.handle_addition(a, b)
    f2 = [a, a5]
    for n in f2:
        a13 = bmg.handle_dot_get(torch, 'tensor')
        a17 = -4.605170249938965
        a9 = bmg.handle_function(a13, [a17])
        a14 = bmg.handle_function(x, [n])
        a6 = bmg.handle_multiplication(a9, a14)
        sum = bmg.handle_addition(sum, a6)
    a10 = 1
    a24 = bmg.handle_dot_get(torch, 'tensor')
    a25 = -0.010050326585769653
    a22 = bmg.handle_function(a24, [a25])
    a20 = bmg.handle_addition(a22, sum)
    a18 = bmg.handle_function(exp, [a20])
    a15 = bmg.handle_negate(a18)
    a7 = bmg.handle_addition(a10, a15)
    r3 = bmg.handle_function(Bernoulli, [a7])
    return bmg.handle_sample(r3)


roots = [z()]
bmg.remove_orphans(roots)
"""

# Demonstrate that function calls work as expected when the
# function called is NOT a sample function.
source5 = """
import torch
from torch.distributions import Bernoulli

# NOTE NO SAMPLE HERE
def q(a, b):
  return (a + b) * 0.5

# NOTE NO SAMPLE HERE
def r(p):
  return Bernoulli(p)

@sample
def x(n):
  return Bernoulli(0.5)

@sample
def z():
  return r(q(x(0), x(1)))
"""

expected_raw_5 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
import torch
from torch.distributions import Bernoulli


def q(a, b):
    a5 = bmg.handle_addition(a, b)
    a8 = 0.5
    r1 = bmg.handle_multiplication(a5, a8)
    return r1


def r(p):
    r2 = bmg.handle_function(Bernoulli, [p])
    return r2


@probabilistic(bmg)
@memoize
def x(n):
    a6 = 0.5
    r3 = bmg.handle_function(Bernoulli, [a6])
    return bmg.handle_sample(r3)


@probabilistic(bmg)
@memoize
def z():
    a10 = 0
    a9 = bmg.handle_function(x, [a10])
    a12 = 1
    a11 = bmg.handle_function(x, [a12])
    a7 = bmg.handle_function(q, [a9, a11])
    r4 = bmg.handle_function(r, [a7])
    return bmg.handle_sample(r4)


roots = [z()]
bmg.remove_orphans(roots)
"""

expected_dot_5 = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label="+"];
  N5[label=0.5];
  N6[label="*"];
  N7[label=Bernoulli];
  N8[label=Sample];
  N1 -> N0[label=probability];
  N2 -> N1[label=operand];
  N3 -> N1[label=operand];
  N4 -> N2[label=left];
  N4 -> N3[label=right];
  N6 -> N4[label=left];
  N6 -> N5[label=right];
  N7 -> N6[label=probability];
  N8 -> N7[label=operand];
}
"""

# Here is a simple model where the argument to a sample is itself a sample.
# This illustrates how the graph must capture the possible control flows.
# Flip a fair coin y; use that to choose which unfair coin to use.
# Flip the unfair coin and use that to construct either a double-headed
# or double-tailed coin.
source6 = """
import torch
from torch.distributions import Bernoulli

# x(0) is Bern(0.25)
# x(1) is Bern(0.75)
@sample
def x(n):
  return Bernoulli(n * 0.5 + 0.25)

@sample
def y():
  return Bernoulli(0.5)

@sample
def z():
  return Bernoulli(x(y()))
"""

expected_raw_6 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
import torch
from torch.distributions import Bernoulli


@probabilistic(bmg)
@memoize
def x(n):
    a9 = 0.5
    a7 = bmg.handle_multiplication(n, a9)
    a10 = 0.25
    a4 = bmg.handle_addition(a7, a10)
    r1 = bmg.handle_function(Bernoulli, [a4])
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def y():
    a5 = 0.5
    r2 = bmg.handle_function(Bernoulli, [a5])
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def z():
    a8 = bmg.handle_function(y, [])
    a6 = bmg.handle_function(x, [a8])
    r3 = bmg.handle_function(Bernoulli, [a6])
    return bmg.handle_sample(r3)


roots = [y(), z()]
bmg.remove_orphans(roots)
"""

expected_dot_6 = """
digraph "graph" {
  N0[label=0.5];
  N10[label=Sample];
  N11[label=map];
  N12[label=index];
  N13[label=Bernoulli];
  N14[label=Sample];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=0.0];
  N4[label=0.25];
  N5[label=Bernoulli];
  N6[label=Sample];
  N7[label=1.0];
  N8[label=0.75];
  N9[label=Bernoulli];
  N1 -> N0[label=probability];
  N10 -> N9[label=operand];
  N11 -> N10[label=3];
  N11 -> N3[label=0];
  N11 -> N6[label=1];
  N11 -> N7[label=2];
  N12 -> N11[label=left];
  N12 -> N2[label=right];
  N13 -> N12[label=probability];
  N14 -> N13[label=operand];
  N2 -> N1[label=operand];
  N5 -> N4[label=probability];
  N6 -> N5[label=operand];
  N9 -> N8[label=probability];
}
"""

# Neal's funnel
source7 = """
from torch.distributions import Normal
from torch import exp

@sample
def X():
  return Normal(0.0, 3.0)

@sample
def Y():
    return Normal(0.0, exp(X() * 0.5))
"""

expected_raw_7 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch.distributions import Normal
from torch import exp


@probabilistic(bmg)
@memoize
def X():
    a3 = 0.0
    a5 = 3.0
    r1 = bmg.handle_function(Normal, [a3, a5])
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def Y():
    a4 = 0.0
    a8 = bmg.handle_function(X, [])
    a9 = 0.5
    a7 = bmg.handle_multiplication(a8, a9)
    a6 = bmg.handle_function(exp, [a7])
    r2 = bmg.handle_function(Normal, [a4, a6])
    return bmg.handle_sample(r2)


roots = [X(), Y()]
bmg.remove_orphans(roots)
"""

expected_dot_7 = """
digraph "graph" {
  N0[label=0.0];
  N1[label=3.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=0.0];
  N5[label=0.5];
  N6[label="*"];
  N7[label=Exp];
  N8[label=Normal];
  N9[label=Sample];
  N2 -> N0[label=mu];
  N2 -> N1[label=sigma];
  N3 -> N2[label=operand];
  N6 -> N3[label=left];
  N6 -> N5[label=right];
  N7 -> N6[label=operand];
  N8 -> N4[label=mu];
  N8 -> N7[label=sigma];
  N9 -> N8[label=operand];
}
"""


class CompilerTest(unittest.TestCase):
    def test_to_python_raw(self) -> None:
        """Tests for to_python_raw from bm_to_bmg.py"""
        self.maxDiff = None
        observed = to_python_raw(source1)
        self.assertEqual(observed.strip(), expected_raw_1.strip())
        observed = to_python_raw(source2)
        self.assertEqual(observed.strip(), expected_raw_2.strip())
        observed = to_python_raw(source3)
        self.assertEqual(observed.strip(), expected_raw_3.strip())
        observed = to_python_raw(source4)
        self.assertEqual(observed.strip(), expected_raw_4.strip())
        observed = to_python_raw(source5)
        self.assertEqual(observed.strip(), expected_raw_5.strip())
        observed = to_python_raw(source6)
        self.assertEqual(observed.strip(), expected_raw_6.strip())
        observed = to_python_raw(source7)
        self.assertEqual(observed.strip(), expected_raw_7.strip())

    def test_to_python(self) -> None:
        """Tests for to_python from bm_to_bmg.py"""
        self.maxDiff = None
        observed = to_python(source1)
        self.assertEqual(observed.strip(), expected_python_1.strip())

    def test_to_dot(self) -> None:
        """Tests for to_dot from bm_to_bmg.py"""
        self.maxDiff = None
        observed = to_dot(source1)
        self.assertEqual(observed.strip(), expected_dot_1.strip())
        observed = to_dot(source2)
        self.assertEqual(observed.strip(), expected_dot_2.strip())
        observed = to_dot(source3)
        self.assertEqual(observed.strip(), expected_dot_3.strip())
        observed = to_dot(source5)
        self.assertEqual(observed.strip(), expected_dot_5.strip())
        observed = to_dot(source6)
        self.assertEqual(observed.strip(), expected_dot_6.strip())
        observed = to_dot(source7)
        self.assertEqual(observed.strip(), expected_dot_7.strip())

    def disabled_test_to_cpp(self) -> None:
        """Tests for to_cpp from bm_to_bmg.py"""
        self.maxDiff = None
        observed = to_cpp(source1)
        self.assertEqual(observed.strip(), expected_cpp_1.strip())

    # TODO: Test disabled; BMG type system violations in graph
    def disabled_test_to_bmg(self) -> None:
        """Tests for to_bmg from bm_to_bmg.py"""
        self.maxDiff = None
        observed = to_bmg(source1).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_1))

    def test_cyclic_model(self) -> None:
        """Tests for to_cpp from bm_to_bmg.py"""

        # The dependency graph here is x -> y -> z -> x
        bad_model_1 = """
import torch
from torch import tensor

@sample
def x():
  return Bernoulli(y())

@sample
def y():
  return Bernoulli(z())

@sample
def z():
  return Bernoulli(x())
"""
        with self.assertRaises(RecursionError):
            to_bmg(bad_model_1)

        # The dependency graph here is z -> x(2) -> y(0) -> x(1) -> y(0)
        bad_model_2 = """
import torch
from torch import tensor

@sample
def x(n):
  return Bernoulli(y(0))

@sample
def y(n):
  return Bernoulli(x(n + 1))

@sample
def z():
  return Bernoulli(x(2))
"""
        with self.assertRaises(RecursionError):
            to_bmg(bad_model_2)
