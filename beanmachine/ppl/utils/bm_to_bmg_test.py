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
        1
        - exp(
            input=log(tensor(0.99))
            + X() * log(tensor(0.01))
            + Y() * log(input=tensor(0.01))
        )
    )

# Verify this is removed
@query
def Q():
  pass
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
    a14 = 0.01
    r11 = [a14]
    a7 = bmg.handle_function(tensor, [*r11], {})
    r4 = [a7]
    r1 = bmg.handle_function(Bernoulli, [*r4], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def Y():
    a15 = 0.01
    r12 = [a15]
    a8 = bmg.handle_function(tensor, [*r12], {})
    r5 = [a8]
    r2 = bmg.handle_function(Bernoulli, [*r5], {})
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def Z():
    a10 = 1
    a21 = bmg.handle_dot_get(torch, 'tensor')
    a29 = -0.010050326585769653
    r26 = [a29]
    a19 = bmg.handle_function(a21, [*r26], {})
    a24 = bmg.handle_function(X, [], {})
    a30 = bmg.handle_dot_get(torch, 'tensor')
    a34 = -4.605170249938965
    r32 = [a34]
    a27 = bmg.handle_function(a30, [*r32], {})
    a22 = bmg.handle_multiplication(a24, a27)
    a18 = bmg.handle_addition(a19, a22)
    a23 = bmg.handle_function(Y, [], {})
    a33 = 0.01
    r31 = [a33]
    a28 = bmg.handle_function(tensor, [*r31], {})
    a25 = bmg.handle_function(log, [], {**{'input': a28}})
    a20 = bmg.handle_multiplication(a23, a25)
    a17 = bmg.handle_addition(a18, a20)
    a16 = bmg.handle_function(exp, [], {**{'input': a17}})
    a13 = bmg.handle_negate(a16)
    a9 = bmg.handle_addition(a10, a13)
    r6 = [a9]
    r3 = bmg.handle_function(Bernoulli, [*r6], {})
    return bmg.handle_sample(r3)


def Q():
    pass


roots = [X(), Y(), Z()]
"""

expected_python_1 = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(tensor(0.009999999776482582))
n1 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n0])
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
n12 = g.add_constant(1.0)
n6 = g.add_constant(tensor(-0.010050326585769653))
n4 = g.add_constant(tensor(-4.605170249938965))
n5 = g.add_operator(graph.OperatorType.MULTIPLY, [n2, n4])
n7 = g.add_operator(graph.OperatorType.ADD, [n6, n5])
n8 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n4])
n9 = g.add_operator(graph.OperatorType.ADD, [n7, n8])
n10 = g.add_operator(graph.OperatorType.EXP, [n9])
n11 = g.add_operator(graph.OperatorType.NEGATE, [n10])
n13 = g.add_operator(graph.OperatorType.ADD, [n12, n11])
n14 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n13])
n15 = g.add_operator(graph.OperatorType.SAMPLE, [n14])
"""

expected_dot_1 = """
digraph "graph" {
  N00[label=0.009999999776482582];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=-4.605170249938965];
  N05[label="*"];
  N06[label=-0.010050326585769653];
  N07[label="+"];
  N08[label="*"];
  N09[label="+"];
  N10[label=Exp];
  N11[label="-"];
  N12[label=1];
  N13[label="+"];
  N14[label=Bernoulli];
  N15[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N03 -> N01[label=operand];
  N05 -> N02[label=left];
  N05 -> N04[label=right];
  N07 -> N05[label=right];
  N07 -> N06[label=left];
  N08 -> N03[label=left];
  N08 -> N04[label=right];
  N09 -> N07[label=left];
  N09 -> N08[label=right];
  N10 -> N09[label=operand];
  N11 -> N10[label=operand];
  N13 -> N11[label=right];
  N13 -> N12[label=left];
  N14 -> N13[label=probability];
  N15 -> N14[label=operand];
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
    return Bernoulli(probs=tensor(0.5) + log(input=exp(input=n * tensor(0.1))))


@sample
def z():
    return Bernoulli(
        tensor(0.3) ** x(0) + x(0) / tensor(10.0) - neg(x(1) * tensor(0.4))
    )
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
    a11 = 0.5
    r8 = [a11]
    a4 = bmg.handle_function(tensor, [*r8], {})
    a25 = 0.1
    r21 = [a25]
    a15 = bmg.handle_function(tensor, [*r21], {})
    a12 = bmg.handle_multiplication(n, a15)
    a9 = bmg.handle_function(exp, [], {**{'input': a12}})
    a6 = bmg.handle_function(log, [], {**{'input': a9}})
    a3 = bmg.handle_addition(a4, a6)
    r1 = bmg.handle_function(Bernoulli, [], {**{'probs': a3}})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def z():
    a26 = 0.3
    r22 = [a26]
    a16 = bmg.handle_function(tensor, [*r22], {})
    a30 = 0
    r27 = [a30]
    a19 = bmg.handle_function(x, [*r27], {})
    a13 = bmg.handle_power(a16, a19)
    a31 = 0
    r28 = [a31]
    a20 = bmg.handle_function(x, [*r28], {})
    a34 = 10.0
    r32 = [a34]
    a23 = bmg.handle_function(tensor, [*r32], {})
    a17 = bmg.handle_division(a20, a23)
    a10 = bmg.handle_addition(a13, a17)
    a37 = 1
    r36 = [a37]
    a33 = bmg.handle_function(x, [*r36], {})
    a39 = 0.4
    r38 = [a39]
    a35 = bmg.handle_function(tensor, [*r38], {})
    a29 = bmg.handle_multiplication(a33, a35)
    r24 = [a29]
    a18 = bmg.handle_function(neg, [*r24], {})
    a14 = bmg.handle_negate(a18)
    a7 = bmg.handle_addition(a10, a14)
    r5 = [a7]
    r2 = bmg.handle_function(Bernoulli, [*r5], {})
    return bmg.handle_sample(r2)


roots = [z()]
"""

expected_dot_2 = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.30000001192092896];
  N04[label="**"];
  N05[label=10.0];
  N06[label="/"];
  N07[label="+"];
  N08[label=0.6000000238418579];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=0.4000000059604645];
  N12[label="*"];
  N13[label="-"];
  N14[label="-"];
  N15[label="+"];
  N16[label=Bernoulli];
  N17[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N02[label=right];
  N04 -> N03[label=left];
  N06 -> N02[label=left];
  N06 -> N05[label=right];
  N07 -> N04[label=left];
  N07 -> N06[label=right];
  N09 -> N08[label=probability];
  N10 -> N09[label=operand];
  N12 -> N10[label=left];
  N12 -> N11[label=right];
  N13 -> N12[label=operand];
  N14 -> N13[label=operand];
  N15 -> N07[label=left];
  N15 -> N14[label=right];
  N16 -> N15[label=probability];
  N17 -> N16[label=operand];
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
    a27 = 0.5
    r25 = [a27]
    a15 = bmg.handle_function(tensor, [*r25], {})
    a40 = 0.1
    r38 = [a40]
    a36 = bmg.handle_function(tensor, [*r38], {})
    a34 = bmg.handle_multiplication(n, a36)
    r32 = [a34]
    a29 = bmg.handle_function(exp, [*r32], {})
    r28 = [a29]
    a21 = bmg.handle_function(log, [*r28], {})
    a13 = bmg.handle_addition(a15, a21)
    r7 = [a13]
    r1 = bmg.handle_function(Bernoulli, [*r7], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def z():
    sum = 0.0
    n = 0
    a8 = bmg.handle_dot_get(torch, 'tensor')
    a22 = -4.605170249938965
    r16 = [a22]
    a5 = bmg.handle_function(a8, [*r16], {})
    r17 = [n]
    a9 = bmg.handle_function(x, [*r17], {})
    a2 = bmg.handle_multiplication(a5, a9)
    sum = bmg.handle_addition(sum, a2)
    n = 1
    a10 = bmg.handle_dot_get(torch, 'tensor')
    a23 = -4.605170249938965
    r18 = [a23]
    a6 = bmg.handle_function(a10, [*r18], {})
    r19 = [n]
    a11 = bmg.handle_function(x, [*r19], {})
    a3 = bmg.handle_multiplication(a6, a11)
    sum = bmg.handle_addition(sum, a3)
    a20 = 1
    a35 = bmg.handle_dot_get(torch, 'tensor')
    a39 = -0.010050326585769653
    r37 = [a39]
    a33 = bmg.handle_function(a35, [*r37], {})
    a31 = bmg.handle_addition(a33, sum)
    r30 = [a31]
    a26 = bmg.handle_function(exp, [*r30], {})
    a24 = bmg.handle_negate(a26)
    a14 = bmg.handle_addition(a20, a24)
    r12 = [a14]
    r4 = bmg.handle_function(Bernoulli, [*r12], {})
    return bmg.handle_sample(r4)


roots = [z()]
"""

expected_dot_3 = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=-4.605170249938965];
  N04[label="*"];
  N05[label=0.0];
  N06[label="+"];
  N07[label=0.6000000238418579];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label="*"];
  N11[label="+"];
  N12[label=-0.010050326585769653];
  N13[label="+"];
  N14[label=Exp];
  N15[label="-"];
  N16[label=1];
  N17[label="+"];
  N18[label=Bernoulli];
  N19[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N02[label=right];
  N04 -> N03[label=left];
  N06 -> N04[label=right];
  N06 -> N05[label=left];
  N08 -> N07[label=probability];
  N09 -> N08[label=operand];
  N10 -> N03[label=left];
  N10 -> N09[label=right];
  N11 -> N06[label=left];
  N11 -> N10[label=right];
  N13 -> N11[label=right];
  N13 -> N12[label=left];
  N14 -> N13[label=operand];
  N15 -> N14[label=operand];
  N17 -> N15[label=right];
  N17 -> N16[label=left];
  N18 -> N17[label=probability];
  N19 -> N18[label=operand];
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
    a22 = 0.5
    r19 = [a22]
    a13 = bmg.handle_function(tensor, [*r19], {})
    a35 = 0.1
    r33 = [a35]
    a31 = bmg.handle_function(tensor, [*r33], {})
    a29 = bmg.handle_multiplication(n, a31)
    r27 = [a29]
    a24 = bmg.handle_function(exp, [*r27], {})
    r23 = [a24]
    a15 = bmg.handle_function(log, [*r23], {})
    a9 = bmg.handle_addition(a13, a15)
    r6 = [a9]
    r1 = bmg.handle_function(Bernoulli, [*r6], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def z():
    sum = 0.0
    a = 2
    b = 3
    a4 = bmg.handle_addition(a, b)
    f2 = [a, a4]
    for n in f2:
        a10 = bmg.handle_dot_get(torch, 'tensor')
        a20 = -4.605170249938965
        r16 = [a20]
        a7 = bmg.handle_function(a10, [*r16], {})
        r17 = [n]
        a11 = bmg.handle_function(x, [*r17], {})
        a5 = bmg.handle_multiplication(a7, a11)
        sum = bmg.handle_addition(sum, a5)
    a14 = 1
    a30 = bmg.handle_dot_get(torch, 'tensor')
    a34 = -0.010050326585769653
    r32 = [a34]
    a28 = bmg.handle_function(a30, [*r32], {})
    a26 = bmg.handle_addition(a28, sum)
    r25 = [a26]
    a21 = bmg.handle_function(exp, [*r25], {})
    a18 = bmg.handle_negate(a21)
    a12 = bmg.handle_addition(a14, a18)
    r8 = [a12]
    r3 = bmg.handle_function(Bernoulli, [*r8], {})
    return bmg.handle_sample(r3)


roots = [z()]
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
    a6 = 0.5
    r1 = bmg.handle_multiplication(a5, a6)
    return r1


def r(p):
    r7 = [p]
    r2 = bmg.handle_function(Bernoulli, [*r7], {})
    return r2


@probabilistic(bmg)
@memoize
def x(n):
    a10 = 0.5
    r8 = [a10]
    r3 = bmg.handle_function(Bernoulli, [*r8], {})
    return bmg.handle_sample(r3)


@probabilistic(bmg)
@memoize
def z():
    a18 = 0
    r17 = [a18]
    a14 = bmg.handle_function(x, [*r17], {})
    a13 = [a14]
    a20 = 1
    r19 = [a20]
    a16 = bmg.handle_function(x, [*r19], {})
    a15 = [a16]
    r12 = bmg.handle_addition(a13, a15)
    a11 = bmg.handle_function(q, [*r12], {})
    r9 = [a11]
    r4 = bmg.handle_function(r, [*r9], {})
    return bmg.handle_sample(r4)


roots = [z()]
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
    a11 = 0.5
    a10 = bmg.handle_multiplication(n, a11)
    a12 = 0.25
    a7 = bmg.handle_addition(a10, a12)
    r4 = [a7]
    r1 = bmg.handle_function(Bernoulli, [*r4], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def y():
    a8 = 0.5
    r5 = [a8]
    r2 = bmg.handle_function(Bernoulli, [*r5], {})
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def z():
    a14 = bmg.handle_function(y, [], {})
    r13 = [a14]
    a9 = bmg.handle_function(x, [*r13], {})
    r6 = [a9]
    r3 = bmg.handle_function(Bernoulli, [*r6], {})
    return bmg.handle_sample(r3)


roots = [y(), z()]
"""

expected_dot_6 = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.25];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=0.0];
  N07[label=0.75];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label=1.0];
  N11[label=map];
  N12[label=index];
  N13[label=Bernoulli];
  N14[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N03[label=probability];
  N05 -> N04[label=operand];
  N08 -> N07[label=probability];
  N09 -> N08[label=operand];
  N11 -> N05[label=1];
  N11 -> N06[label=0];
  N11 -> N09[label=3];
  N11 -> N10[label=2];
  N12 -> N02[label=right];
  N12 -> N11[label=left];
  N13 -> N12[label=probability];
  N14 -> N13[label=operand];
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
    return Normal(loc=0.0, scale=exp(X() * 0.5))
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
    a9 = 0.0
    a7 = [a9]
    a12 = 3.0
    a10 = [a12]
    r5 = bmg.handle_addition(a7, a10)
    r1 = bmg.handle_function(Normal, [*r5], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def Y():
    a3 = 0.0
    a11 = bmg.handle_function(X, [], {})
    a13 = 0.5
    a8 = bmg.handle_multiplication(a11, a13)
    r6 = [a8]
    a4 = bmg.handle_function(exp, [*r6], {})
    r2 = bmg.handle_function(Normal, [], {**{'loc': a3}, **{'scale': a4}})
    return bmg.handle_sample(r2)


roots = [X(), Y()]
"""

expected_dot_7 = """
digraph "graph" {
  N0[label=0.0];
  N1[label=3.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=0.5];
  N5[label="*"];
  N6[label=Exp];
  N7[label=0.0];
  N8[label=Normal];
  N9[label=Sample];
  N2 -> N0[label=mu];
  N2 -> N1[label=sigma];
  N3 -> N2[label=operand];
  N5 -> N3[label=left];
  N5 -> N4[label=right];
  N6 -> N5[label=operand];
  N8 -> N6[label=sigma];
  N8 -> N7[label=mu];
  N9 -> N8[label=operand];
}
"""

# Mint an unfair coin, then flip it.
source8 = """
from torch.distributions import Beta, Bernoulli

# What is the unfairness of the coins from this mint?
@sample
def mint():
    return Beta(1.0, 1.0)

# Mint a coin and then toss it.
@sample
def toss():
    return Bernoulli(mint())
"""

expected_raw_8 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch.distributions import Beta, Bernoulli


@probabilistic(bmg)
@memoize
def mint():
    a7 = 1.0
    a6 = [a7]
    a9 = 1.0
    a8 = [a9]
    r5 = bmg.handle_addition(a6, a8)
    r1 = bmg.handle_function(Beta, [*r5], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def toss():
    a4 = bmg.handle_function(mint, [], {})
    r3 = [a4]
    r2 = bmg.handle_function(Bernoulli, [*r3], {})
    return bmg.handle_sample(r2)


roots = [mint(), toss()]
"""

expected_dot_8 = """
digraph "graph" {
  N0[label=1.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Bernoulli];
  N4[label=Sample];
  N1 -> N0[label=alpha];
  N1 -> N0[label=beta];
  N2 -> N1[label=operand];
  N3 -> N2[label=probability];
  N4 -> N3[label=operand];
}
"""

# Testing support for calls with keyword args
source9 = """
from torch.distributions import Bernoulli

@sample
def toss():
    return Bernoulli(probs=0.5)

# Notice that logits Bernoulli with constant argument is folded to
# probs Bernoulli...
@sample
def toss2():
    return Bernoulli(logits=0.0)

# ...but we must make a distinction between logits and probs if the
# argument is a sample.
@sample
def toss3():
    return Bernoulli(probs=toss())

@sample
def toss4():
    return Bernoulli(logits=toss())
"""

expected_raw_9 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch.distributions import Bernoulli


@probabilistic(bmg)
@memoize
def toss():
    a5 = 0.5
    r1 = bmg.handle_function(Bernoulli, [], {**{'probs': a5}})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def toss2():
    a6 = 0.0
    r2 = bmg.handle_function(Bernoulli, [], {**{'logits': a6}})
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def toss3():
    a7 = bmg.handle_function(toss, [], {})
    r3 = bmg.handle_function(Bernoulli, [], {**{'probs': a7}})
    return bmg.handle_sample(r3)


@probabilistic(bmg)
@memoize
def toss4():
    a8 = bmg.handle_function(toss, [], {})
    r4 = bmg.handle_function(Bernoulli, [], {**{'logits': a8}})
    return bmg.handle_sample(r4)


roots = [toss(), toss2(), toss3(), toss4()]
"""

expected_dot_9 = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Sample];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label="Bernoulli(logits)"];
  N7[label=Sample];
  N1 -> N0[label=probability];
  N2 -> N1[label=operand];
  N3 -> N1[label=operand];
  N4 -> N2[label=probability];
  N5 -> N4[label=operand];
  N6 -> N2[label=probability];
  N7 -> N6[label=operand];
}
"""

# Bayesian regression
source10 = """
from torch import tensor, zeros
from torch.distributions import Normal, Bernoulli
N = 3
K = 2
X = tensor([[1.0, 10, 20], [1.0, -100, -190], [1.0, -101, -192]])
intercept_scale = 0.9
coef_scale = [1.2, 2.3]

@sample
def beta():
    return Normal(
        zeros((K + 1, 1)), tensor([intercept_scale] + coef_scale).view(K + 1, 1)
    )

@sample
def y():
    return Bernoulli(logits=X.mm(beta()))
"""

expected_raw_10 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch import tensor, zeros
from torch.distributions import Normal, Bernoulli
N = 3
K = 2
a9 = 1.0
a14 = 10
a19 = 20
a5 = [a9, a14, a19]
a15 = 1.0
a20 = -100
a24 = -190
a10 = [a15, a20, a24]
a21 = 1.0
a25 = -101
a29 = -192
a16 = [a21, a25, a29]
a1 = [a5, a10, a16]
X = bmg.handle_function(tensor, [a1], {})
intercept_scale = 0.9
a2 = 1.2
a6 = 2.3
coef_scale = [a2, a6]


@probabilistic(bmg)
@memoize
def beta():
    a11 = K + 1, 1
    a7 = bmg.handle_function(zeros, [a11], {})
    a30 = [intercept_scale]
    a26 = bmg.handle_addition(a30, coef_scale)
    a22 = bmg.handle_function(tensor, [a26], {})
    a17 = bmg.handle_dot_get(a22, 'view')
    a27 = 1
    a23 = bmg.handle_addition(K, a27)
    a28 = 1
    a12 = bmg.handle_function(a17, [a23, a28], {})
    r3 = bmg.handle_function(Normal, [a7, a12], {})
    return bmg.handle_sample(r3)


@probabilistic(bmg)
@memoize
def y():
    a13 = bmg.handle_dot_get(X, 'mm')
    a18 = bmg.handle_function(beta, [], {})
    a8 = bmg.handle_function(a13, [a18], {})
    r4 = bmg.handle_function(Bernoulli, [], {**{'logits': a8}})
    return bmg.handle_sample(r4)


roots = [beta(), y()]
"""

expected_dot_10 = """
digraph "graph" {
  N0[label="[[0.0],\\\\n[0.0],\\\\n[0.0]]"];
  N1[label="[[0.8999999761581421],\\\\n[1.2000000476837158],\\\\n[2.299999952316284]]"];
  N2[label=Normal];
  N3[label=Sample];
  N4[label="[[1.0,10.0,20.0],\\\\n[1.0,-100.0,-190.0],\\\\n[1.0,-101.0,-192.0]]"];
  N5[label="*"];
  N6[label="Bernoulli(logits)"];
  N7[label=Sample];
  N2 -> N0[label=mu];
  N2 -> N1[label=sigma];
  N3 -> N2[label=operand];
  N5 -> N3[label=right];
  N5 -> N4[label=left];
  N6 -> N5[label=probability];
  N7 -> N6[label=operand];
}
"""

# A sketch of a model for predicting if a new account is fake based on
# friend requests issued and accepted.
source11 = """
from torch import tensor
from torch.distributions import Bernoulli
FAKE_PRIOR = 0.001
# One entry per user
FAKE_REQ_PROB = tensor([0.01, 0.02, 0.03])
REAL_REQ_PROB = tensor([0.04, 0.05, 0.06])
REQ_PROB = [REAL_REQ_PROB, FAKE_REQ_PROB]
REAL_ACC_PROB = tensor([0.99, 0.50, 0.07])
@sample
def is_fake(account):
  return Bernoulli(FAKE_PRIOR)
@sample
def all_requests_sent(account):
  return Bernoulli(REQ_PROB[is_fake(account)])
@sample
def all_requests_accepted(account):
  return Bernoulli(REAL_ACC_PROB * all_requests_sent(account))
_1 = 0
_2 = all_requests_accepted(_1)
"""

expected_raw_11 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg: bool = True
bmg = BMGraphBuilder()
from torch import tensor
from torch.distributions import Bernoulli
FAKE_PRIOR = 0.001
a14 = 0.01
a19 = 0.02
a24 = 0.03
a8 = [a14, a19, a24]
r4 = [a8]
FAKE_REQ_PROB = bmg.handle_function(tensor, [*r4], {})
a15 = 0.04
a20 = 0.05
a25 = 0.06
a9 = [a15, a20, a25]
r5 = [a9]
REAL_REQ_PROB = bmg.handle_function(tensor, [*r5], {})
REQ_PROB = [REAL_REQ_PROB, FAKE_REQ_PROB]
a16 = 0.99
a21 = 0.5
a26 = 0.07
a10 = [a16, a21, a26]
r6 = [a10]
REAL_ACC_PROB = bmg.handle_function(tensor, [*r6], {})


@probabilistic(bmg)
@memoize
def is_fake(account):
    r11 = [FAKE_PRIOR]
    r1 = bmg.handle_function(Bernoulli, [*r11], {})
    return bmg.handle_sample(r1)


@probabilistic(bmg)
@memoize
def all_requests_sent(account):
    r27 = [account]
    a22 = bmg.handle_function(is_fake, [*r27], {})
    a17 = bmg.handle_index(REQ_PROB, a22)
    r12 = [a17]
    r2 = bmg.handle_function(Bernoulli, [*r12], {})
    return bmg.handle_sample(r2)


@probabilistic(bmg)
@memoize
def all_requests_accepted(account):
    r28 = [account]
    a23 = bmg.handle_function(all_requests_sent, [*r28], {})
    a18 = bmg.handle_multiplication(REAL_ACC_PROB, a23)
    r13 = [a18]
    r3 = bmg.handle_function(Bernoulli, [*r13], {})
    return bmg.handle_sample(r3)


_1 = 0
r7 = [_1]
_2 = bmg.handle_function(all_requests_accepted, [*r7], {})
roots = []
"""

expected_dot_11 = """
digraph "graph" {
  N00[label=0.0010000000474974513];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0];
  N04[label="[0.03999999910593033,0.05000000074505806,0.05999999865889549]"];
  N05[label=1];
  N06[label="[0.009999999776482582,0.019999999552965164,0.029999999329447746]"];
  N07[label=map];
  N08[label=index];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label="[0.9900000095367432,0.5,0.07000000029802322]"];
  N12[label="*"];
  N13[label=Bernoulli];
  N14[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N07 -> N03[label=0];
  N07 -> N04[label=1];
  N07 -> N05[label=2];
  N07 -> N06[label=3];
  N08 -> N02[label=right];
  N08 -> N07[label=left];
  N09 -> N08[label=probability];
  N10 -> N09[label=operand];
  N12 -> N10[label=right];
  N12 -> N11[label=left];
  N13 -> N12[label=probability];
  N14 -> N13[label=operand];
}
"""

source12 = """
from torch.distributions import Normal, Uniform
@sample
def theta_0():
    return Normal(0,1)

@sample
def theta_1():
    return Normal(0,1)

@sample
def error():
    return Uniform(0,1)

@sample
def x(i):
    return Normal(0,1)

@sample
def y(i):
    return Normal(theta_0() + theta_1() * x(i), error())

observations = [y(i) for i in range(3)] + [x(i) for i in range(3)]
"""

expected_dot_12 = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label="*"];
  N07[label="+"];
  N08[label=Uniform];
  N09[label=Sample];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=Sample];
  N13[label="*"];
  N14[label="+"];
  N15[label=Normal];
  N16[label=Sample];
  N17[label=Sample];
  N18[label="*"];
  N19[label="+"];
  N20[label=Normal];
  N21[label=Sample];
  N02 -> N00[label=mu];
  N02 -> N01[label=sigma];
  N03 -> N02[label=operand];
  N04 -> N02[label=operand];
  N05 -> N02[label=operand];
  N06 -> N04[label=left];
  N06 -> N05[label=right];
  N07 -> N03[label=left];
  N07 -> N06[label=right];
  N08 -> N00[label=low];
  N08 -> N01[label=high];
  N09 -> N08[label=operand];
  N10 -> N07[label=mu];
  N10 -> N09[label=sigma];
  N11 -> N10[label=operand];
  N12 -> N02[label=operand];
  N13 -> N04[label=left];
  N13 -> N12[label=right];
  N14 -> N03[label=left];
  N14 -> N13[label=right];
  N15 -> N09[label=sigma];
  N15 -> N14[label=mu];
  N16 -> N15[label=operand];
  N17 -> N02[label=operand];
  N18 -> N04[label=left];
  N18 -> N17[label=right];
  N19 -> N03[label=left];
  N19 -> N18[label=right];
  N20 -> N09[label=sigma];
  N20 -> N19[label=mu];
  N21 -> N20[label=operand];
}
"""

# Illustrate that we correctly generate the support for
# multidimensional Bernoulli distributions. Flip two coins,
# take their average, and use that to make a third coin:
source13 = """
import torch
from torch import tensor
from torch.distributions import Bernoulli

@sample
def x(n):
  return Bernoulli(n.sum()*0.5)

@sample
def y():
  return Bernoulli(tensor([0.5,0.5]))

@sample
def z():
  return Bernoulli(x(y()))
"""

expected_dot_13 = """
digraph "graph" {
  N00[label="[0.5,0.5]"];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label="[0.0,0.0]"];
  N07[label=0.5];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label="[0.0,1.0]"];
  N11[label=Sample];
  N12[label="[1.0,0.0]"];
  N13[label=1.0];
  N14[label=Bernoulli];
  N15[label=Sample];
  N16[label="[1.0,1.0]"];
  N17[label=map];
  N18[label=index];
  N19[label=Bernoulli];
  N20[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N03[label=probability];
  N05 -> N04[label=operand];
  N08 -> N07[label=probability];
  N09 -> N08[label=operand];
  N11 -> N08[label=operand];
  N14 -> N13[label=probability];
  N15 -> N14[label=operand];
  N17 -> N05[label=1];
  N17 -> N06[label=0];
  N17 -> N09[label=3];
  N17 -> N10[label=2];
  N17 -> N11[label=5];
  N17 -> N12[label=4];
  N17 -> N15[label=7];
  N17 -> N16[label=6];
  N18 -> N02[label=right];
  N18 -> N17[label=left];
  N19 -> N18[label=probability];
  N20 -> N19[label=operand];
}
"""

# Simple example of categorical
source14 = """
import torch
from torch.distributions import Bernoulli, Categorical
from torch import tensor

@sample
def x(n):
  if n == 0:
    return Bernoulli(0.5)
  if n == 1:
    return Categorical(tensor([1.0, 3.0, 4.0]))
  return Bernoulli(0.75)

@sample
def y():
  return Categorical(tensor([2.0, 6.0, 8.0]))

@sample
def z():
  p = x(y()) * 0.25
  return Bernoulli(p)
"""

expected_dot_14 = """
digraph "graph" {
  N00[label="[0.125,0.375,0.5]"];
  N01[label=Categorical];
  N02[label=Sample];
  N03[label=0.5];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=0];
  N07[label=Sample];
  N08[label=1];
  N09[label=0.75];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=2];
  N13[label=map];
  N14[label=index];
  N15[label=0.25];
  N16[label="*"];
  N17[label=Bernoulli];
  N18[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N03[label=probability];
  N05 -> N04[label=operand];
  N07 -> N01[label=operand];
  N10 -> N09[label=probability];
  N11 -> N10[label=operand];
  N13 -> N05[label=1];
  N13 -> N06[label=0];
  N13 -> N07[label=3];
  N13 -> N08[label=2];
  N13 -> N11[label=5];
  N13 -> N12[label=4];
  N14 -> N02[label=right];
  N14 -> N13[label=left];
  N16 -> N14[label=left];
  N16 -> N15[label=right];
  N17 -> N16[label=probability];
  N18 -> N17[label=operand];
}
"""

# Gaussian mixture model.  Suppose we have a mixture of k normal distributions
# each with standard deviation equal to 1, but different means. Our prior
# on means is that mu(0), ... mu(k) are normally distributed.
# To make samples y(0), ... from this distribution we first choose which
# mean we want with z(0), ..., use that to sample mu(z(0)) to get the mean,
# and then use that mean to sample from a normal distribution.
source15 = """
from torch import tensor
from torch.distributions import Categorical, Normal

@sample
def mu(k):
    # Means of the components are normally distributed
    return Normal(0, 1)

@sample
def z(i):
    # Choose a category, 0, 1 or 2 with ratio 1:3:4.
    return Categorical(tensor([1., 3., 4.]))

@sample
def y(i):
    return Normal(mu(z(i)), 1)

y0 = y(0)
"""

expected_dot_15 = """
digraph "graph" {
  N00[label="[0.125,0.375,0.5]"];
  N01[label=Categorical];
  N02[label=Sample];
  N03[label=0.0];
  N04[label=1.0];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=0];
  N08[label=Sample];
  N09[label=1];
  N10[label=Sample];
  N11[label=2];
  N12[label=map];
  N13[label=index];
  N14[label=1];
  N15[label=Normal];
  N16[label=Sample];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N05 -> N03[label=mu];
  N05 -> N04[label=sigma];
  N06 -> N05[label=operand];
  N08 -> N05[label=operand];
  N10 -> N05[label=operand];
  N12 -> N06[label=1];
  N12 -> N07[label=0];
  N12 -> N08[label=3];
  N12 -> N09[label=2];
  N12 -> N10[label=5];
  N12 -> N11[label=4];
  N13 -> N02[label=right];
  N13 -> N12[label=left];
  N15 -> N13[label=mu];
  N15 -> N14[label=sigma];
  N16 -> N15[label=operand];
}
"""

# Simplified CLARA model
source16 = """
from torch import tensor
from torch.distributions import Categorical, Dirichlet
# three categories
sheep = 0
goat = 1
cat = 2
# two classifiers
alice = 10
bob = 11
# two photos
foo_jpg = 20
bar_jpg = 21

# Prior on distribution of true distribution of categories
# amongst documents; no distribution is more likely than
# any other.

beta = tensor([1., 1., 1.])

@sample
def pi():
  return Dirichlet(beta)

# Prior on confusion of classifiers; classifiers confuse sheep with
# goats more easily than sheep with cats or goats with cats:
alpha = [
  # Given a sheep, we are unlikely to classify it as a cat
  tensor([10., 5., 1.]),
  # Given a goat, we are unlikely to classify it as a cat
  tensor([5., 10., 1.]),
  # Given a cat, we are likely to classify it as a cat
  tensor([1., 1., 10.]),
]

# For each classifier j, and each category k, sample a categorical distribution
# of likely classifications of an item truly of category k.
@sample
def theta(j: int, k: int):
    return Dirichlet(alpha[k])

# For each item i, sample it from a collection of items whose true
# categories are distributed by pi. The output sample is a category.
@sample
def z(i: int):
    return Categorical(pi())

# We then simulate classification of item i by classifier j.
# Given true category z(i) for the item i, the behaviour of
# classifier j will be to sample from a categorical distribution
# whose parameters are theta(j, z(i))
# The output sample is again a category.
@sample
def y(i: int, j: int):
    return Categorical(theta(j, z(i)))

observations = {
  y(foo_jpg, alice): sheep,
  y(foo_jpg, bob): goat,
  y(bar_jpg, alice): cat,
  y(bar_jpg, bob): cat,
}
"""

expected_dot_16 = """
digraph "graph" {
  N00[label="[1.0,1.0,1.0]"];
  N01[label=Dirichlet];
  N02[label=Sample];
  N03[label=Categorical];
  N04[label=Sample];
  N05[label="[10.0,5.0,1.0]"];
  N06[label=Dirichlet];
  N07[label=Sample];
  N08[label=0];
  N09[label="[5.0,10.0,1.0]"];
  N10[label=Dirichlet];
  N11[label=Sample];
  N12[label=1];
  N13[label="[1.0,1.0,10.0]"];
  N14[label=Dirichlet];
  N15[label=Sample];
  N16[label=2];
  N17[label=map];
  N18[label=index];
  N19[label=Categorical];
  N20[label=Sample];
  N21[label=Sample];
  N22[label=Sample];
  N23[label=Sample];
  N24[label=map];
  N25[label=index];
  N26[label=Categorical];
  N27[label=Sample];
  N28[label=Sample];
  N29[label=index];
  N30[label=Categorical];
  N31[label=Sample];
  N32[label=index];
  N33[label=Categorical];
  N34[label=Sample];
  N01 -> N00[label=concentration];
  N02 -> N01[label=operand];
  N03 -> N02[label=probability];
  N04 -> N03[label=operand];
  N06 -> N05[label=concentration];
  N07 -> N06[label=operand];
  N10 -> N09[label=concentration];
  N11 -> N10[label=operand];
  N14 -> N13[label=concentration];
  N15 -> N14[label=operand];
  N17 -> N07[label=1];
  N17 -> N08[label=0];
  N17 -> N11[label=3];
  N17 -> N12[label=2];
  N17 -> N15[label=5];
  N17 -> N16[label=4];
  N18 -> N04[label=right];
  N18 -> N17[label=left];
  N19 -> N18[label=probability];
  N20 -> N19[label=operand];
  N21 -> N06[label=operand];
  N22 -> N10[label=operand];
  N23 -> N14[label=operand];
  N24 -> N08[label=0];
  N24 -> N12[label=2];
  N24 -> N16[label=4];
  N24 -> N21[label=1];
  N24 -> N22[label=3];
  N24 -> N23[label=5];
  N25 -> N04[label=right];
  N25 -> N24[label=left];
  N26 -> N25[label=probability];
  N27 -> N26[label=operand];
  N28 -> N03[label=operand];
  N29 -> N17[label=left];
  N29 -> N28[label=right];
  N30 -> N29[label=probability];
  N31 -> N30[label=operand];
  N32 -> N24[label=left];
  N32 -> N28[label=right];
  N33 -> N32[label=probability];
  N34 -> N33[label=operand];
}
"""

# Bayesian Meta-analysis example
source17 = """
from torch.distributions import HalfCauchy, Normal, StudentT
from torch import tensor
from beanmachine.ppl.model.statistical_model import sample

class Node:
    def __init__(self, level=None, parent=None, result=None, stddev=None):
      if level is None:
        self.level = parent.level - 1
      else:
        self.level = level
      self.parent = parent
      self.result = result
      self.stddev = stddev

group1 = Node(level=2)
team1 = Node(parent=group1)
team2 = Node(parent=group1)
group2 = Node(level=2)
team3 = Node(parent=group2)
team4 = Node(parent=group2)
experiments = [
    Node(result=19.8,  stddev=3.1, parent=team1),
    Node(result=-12.5, stddev=3.4, parent=team1),
    Node(result=52.7,  stddev=7.4, parent=team2),
    Node(result=57.3,  stddev=4.2, parent=team2),
    Node(result=-61.5, stddev=2.3, parent=team3),
    Node(result=-16.9, stddev=3.1, parent=team3),
    Node(result=15.3,  stddev=3.1, parent=team4),
    Node(result=32.3,  stddev=3.2, parent=team4),
]

@sample
def experiment_result(experiment):
    mean = (
        true_value() +
        # Omitted to make the graph easier to read
        # node_bias(experiment) +
        node_bias(experiment.parent) +
        node_bias(experiment.parent.parent))
    return Normal(mean, experiment.stddev)

@sample
def true_value():
    return StudentT(3, 0, 10)

@sample
def node_bias(node):
    return Normal(0, sigma(node.level))

@sample
def sigma(level):
    return HalfCauchy(tensor(1.))

for x in experiments:
    experiment_result(x)
"""

expected_dot_17 = """
digraph "graph" {
  N00[label=3.0];
  N01[label=0.0];
  N02[label=10.0];
  N03[label=StudentT];
  N04[label=Sample];
  N05[label=1.0];
  N06[label=HalfCauchy];
  N07[label=Sample];
  N08[label=0];
  N09[label=Normal];
  N10[label=Sample];
  N11[label="+"];
  N12[label=Sample];
  N13[label=Normal];
  N14[label=Sample];
  N15[label="+"];
  N16[label=3.1];
  N17[label=Normal];
  N18[label=Sample];
  N19[label=3.4];
  N20[label=Normal];
  N21[label=Sample];
  N22[label=Sample];
  N23[label="+"];
  N24[label="+"];
  N25[label=7.4];
  N26[label=Normal];
  N27[label=Sample];
  N28[label=4.2];
  N29[label=Normal];
  N30[label=Sample];
  N31[label=Sample];
  N32[label="+"];
  N33[label=Sample];
  N34[label="+"];
  N35[label=2.3];
  N36[label=Normal];
  N37[label=Sample];
  N38[label=Normal];
  N39[label=Sample];
  N40[label=Sample];
  N41[label="+"];
  N42[label="+"];
  N43[label=Normal];
  N44[label=Sample];
  N45[label=3.2];
  N46[label=Normal];
  N47[label=Sample];
  N03 -> N00[label=df];
  N03 -> N01[label=loc];
  N03 -> N02[label=scale];
  N04 -> N03[label=operand];
  N06 -> N05[label=scale];
  N07 -> N06[label=operand];
  N09 -> N07[label=sigma];
  N09 -> N08[label=mu];
  N10 -> N09[label=operand];
  N11 -> N04[label=left];
  N11 -> N10[label=right];
  N12 -> N06[label=operand];
  N13 -> N08[label=mu];
  N13 -> N12[label=sigma];
  N14 -> N13[label=operand];
  N15 -> N11[label=left];
  N15 -> N14[label=right];
  N17 -> N15[label=mu];
  N17 -> N16[label=sigma];
  N18 -> N17[label=operand];
  N20 -> N15[label=mu];
  N20 -> N19[label=sigma];
  N21 -> N20[label=operand];
  N22 -> N09[label=operand];
  N23 -> N04[label=left];
  N23 -> N22[label=right];
  N24 -> N14[label=right];
  N24 -> N23[label=left];
  N26 -> N24[label=mu];
  N26 -> N25[label=sigma];
  N27 -> N26[label=operand];
  N29 -> N24[label=mu];
  N29 -> N28[label=sigma];
  N30 -> N29[label=operand];
  N31 -> N09[label=operand];
  N32 -> N04[label=left];
  N32 -> N31[label=right];
  N33 -> N13[label=operand];
  N34 -> N32[label=left];
  N34 -> N33[label=right];
  N36 -> N34[label=mu];
  N36 -> N35[label=sigma];
  N37 -> N36[label=operand];
  N38 -> N16[label=sigma];
  N38 -> N34[label=mu];
  N39 -> N38[label=operand];
  N40 -> N09[label=operand];
  N41 -> N04[label=left];
  N41 -> N40[label=right];
  N42 -> N33[label=right];
  N42 -> N41[label=left];
  N43 -> N16[label=sigma];
  N43 -> N42[label=mu];
  N44 -> N43[label=operand];
  N46 -> N42[label=mu];
  N46 -> N45[label=sigma];
  N47 -> N46[label=operand];
}
"""


source18 = """
from torch.distributions import HalfCauchy, Normal, StudentT
from torch import tensor
from beanmachine.ppl.model.statistical_model import query, sample

@sample
def result():
    return Normal(mean(), 1.0)

@query
def mean():
  return true_value() + bias()

@sample
def true_value():
    return StudentT(3, 0, 10)

@sample
def bias():
    return Normal(0, sigma())

@sample
def sigma():
    return HalfCauchy(tensor(1.))
"""

expected_dot_18 = """
digraph "graph" {
  N00[label=3.0];
  N01[label=0.0];
  N02[label=10.0];
  N03[label=StudentT];
  N04[label=Sample];
  N05[label=1.0];
  N06[label=HalfCauchy];
  N07[label=Sample];
  N08[label=0];
  N09[label=Normal];
  N10[label=Sample];
  N11[label="+"];
  N12[label=Query];
  N13[label=1.0];
  N14[label=Normal];
  N15[label=Sample];
  N03 -> N00[label=df];
  N03 -> N01[label=loc];
  N03 -> N02[label=scale];
  N04 -> N03[label=operand];
  N06 -> N05[label=scale];
  N07 -> N06[label=operand];
  N09 -> N07[label=sigma];
  N09 -> N08[label=mu];
  N10 -> N09[label=operand];
  N11 -> N04[label=left];
  N11 -> N10[label=right];
  N12 -> N11[label=operator];
  N14 -> N11[label=mu];
  N14 -> N13[label=sigma];
  N15 -> N14[label=operand];
}
"""


class CompilerTest(unittest.TestCase):
    def disabled_test_to_python_raw(self) -> None:
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
        observed = to_python_raw(source8)
        self.assertEqual(observed.strip(), expected_raw_8.strip())
        observed = to_python_raw(source9)
        self.assertEqual(observed.strip(), expected_raw_9.strip())
        #        observed = to_python_raw(source10)
        #        self.assertEqual(observed.strip(), expected_raw_10.strip())
        observed = to_python_raw(source11)
        self.assertEqual(observed.strip(), expected_raw_11.strip())

    def disabled_test_to_python(self) -> None:
        """Tests for to_python from bm_to_bmg.py"""
        self.maxDiff = None
        # TODO: This test is disabled because the model computes a probability
        # TODO: via addition, which is not supported in the BMG type system.
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
        observed = to_dot(source8)
        self.assertEqual(observed.strip(), expected_dot_8.strip())
        observed = to_dot(source9)
        self.assertEqual(observed.strip(), expected_dot_9.strip())
        observed = to_dot(source10)
        self.assertEqual(observed.strip(), expected_dot_10.strip())
        observed = to_dot(source11)
        self.assertEqual(observed.strip(), expected_dot_11.strip())
        observed = to_dot(source12)
        self.assertEqual(observed.strip(), expected_dot_12.strip())
        observed = to_dot(source13)
        self.assertEqual(observed.strip(), expected_dot_13.strip())
        observed = to_dot(source14)
        self.assertEqual(observed.strip(), expected_dot_14.strip())
        observed = to_dot(source15)
        self.assertEqual(observed.strip(), expected_dot_15.strip())
        observed = to_dot(source16)
        self.assertEqual(observed.strip(), expected_dot_16.strip())
        observed = to_dot(source17)
        self.assertEqual(observed.strip(), expected_dot_17.strip())
        observed = to_dot(source18)
        self.assertEqual(observed.strip(), expected_dot_18.strip())

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
