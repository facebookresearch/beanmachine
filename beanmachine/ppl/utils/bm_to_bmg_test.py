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


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


source1 = """
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli

@sample
def X():
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
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
bmg = BMGraphBuilder()
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli


@memoize
def X():
    a4 = bmg.add_tensor(tensor(0.01))
    r1 = bmg.add_bernoulli(bmg.add_to_real(a4))
    return bmg.add_sample(r1)


@memoize
def Y():
    a5 = bmg.add_tensor(tensor(0.01))
    r2 = bmg.add_bernoulli(bmg.add_to_real(a5))
    return bmg.add_sample(r2)


@memoize
def Z():
    a7 = bmg.add_real(1)
    a12 = bmg.add_tensor(torch.tensor(-0.010050326585769653))
    a16 = X()
    a18 = bmg.add_tensor(torch.tensor(-4.605170249938965))
    a14 = bmg.add_multiplication(a16, a18)
    a11 = bmg.add_addition(a12, a14)
    a15 = Y()
    a17 = bmg.add_tensor(torch.tensor(-4.605170249938965))
    a13 = bmg.add_multiplication(a15, a17)
    a10 = bmg.add_addition(a11, a13)
    a9 = bmg.add_exp(a10)
    a8 = bmg.add_negate(a9)
    a6 = bmg.add_addition(a7, a8)
    r3 = bmg.add_bernoulli(bmg.add_to_real(a6))
    return bmg.add_sample(r3)


roots = [X(), Y(), Z()]
bmg.remove_orphans(roots)
"""

expected_python_1 = (
    """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(0.009999999776482582)
n1 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n0])"""  # noqa: B950
    + """
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
n14 = g.add_operator(graph.OperatorType.TO_REAL, [n13])
n15 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n14])"""  # noqa: B950
    + """
n16 = g.add_operator(graph.OperatorType.SAMPLE, [n15])
"""
)

expected_dot_1 = """
digraph "graph" {
  N0[label=0.009999999776482582];
  N10[label="+"];
  N11[label=Exp];
  N12[label="-"];
  N13[label="+"];
  N14[label=ToReal];
  N15[label=Bernoulli];
  N16[label=Sample];
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
  N14 -> N13[label=operand];
  N15 -> N14[label=probability];
  N16 -> N15[label=operand];
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
from torch import tensor

@sample
def x(n):
  return Bernoulli(tensor(0.5) + n * tensor(0.1))

@sample
def z():
  return Bernoulli(x(0) * tensor(0.3) + x(0) * tensor(0.1) + x(1) * tensor(0.4))
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
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
bmg = BMGraphBuilder()
import torch
from torch import tensor


@memoize
def x(n):
    a5 = bmg.add_tensor(tensor(0.5))
    a10 = bmg.add_tensor(tensor(0.1))
    a7 = bmg.add_multiplication(n, a10)
    a3 = bmg.add_addition(a5, a7)
    r1 = bmg.add_bernoulli(bmg.add_to_real(a3))
    return bmg.add_sample(r1)


@memoize
def z():
    a14 = bmg.add_real(0)
    a11 = x(a14)
    a15 = bmg.add_tensor(tensor(0.3))
    a8 = bmg.add_multiplication(a11, a15)
    a19 = bmg.add_real(0)
    a16 = x(a19)
    a20 = bmg.add_tensor(tensor(0.1))
    a12 = bmg.add_multiplication(a16, a20)
    a6 = bmg.add_addition(a8, a12)
    a17 = bmg.add_real(1)
    a13 = x(a17)
    a18 = bmg.add_tensor(tensor(0.4))
    a9 = bmg.add_multiplication(a13, a18)
    a4 = bmg.add_addition(a6, a9)
    r2 = bmg.add_bernoulli(bmg.add_to_real(a4))
    return bmg.add_sample(r2)


roots = [z()]
bmg.remove_orphans(roots)
"""

expected_dot_2 = """
digraph "graph" {
  N0[label=0.5];
  N10[label=Sample];
  N11[label=0.4000000059604645];
  N12[label="*"];
  N13[label="+"];
  N14[label=ToReal];
  N15[label=Bernoulli];
  N16[label=Sample];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=0.30000001192092896];
  N4[label="*"];
  N5[label=0.10000000149011612];
  N6[label="*"];
  N7[label="+"];
  N8[label=0.6000000238418579];
  N9[label=Bernoulli];
  N1 -> N0[label=probability];
  N10 -> N9[label=operand];
  N12 -> N10[label=left];
  N12 -> N11[label=right];
  N13 -> N12[label=right];
  N13 -> N7[label=left];
  N14 -> N13[label=operand];
  N15 -> N14[label=probability];
  N16 -> N15[label=operand];
  N2 -> N1[label=operand];
  N4 -> N2[label=left];
  N4 -> N3[label=right];
  N6 -> N2[label=left];
  N6 -> N5[label=right];
  N7 -> N4[label=left];
  N7 -> N6[label=right];
  N9 -> N8[label=probability];
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

    def test_to_cpp(self) -> None:
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
