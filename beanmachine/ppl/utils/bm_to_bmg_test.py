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


X()
Y()
Z()
"""

expected_python_1 = (
    """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(tensor(0.009999999776482582))
n1 = g.add_operator(graph.OperatorType.TO_REAL, [n0])
n2 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n1])"""  # noqa: B950
    + """
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_constant(tensor(0.009999999776482582))
n5 = g.add_operator(graph.OperatorType.TO_REAL, [n4])
n6 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n5])"""  # noqa: B950
    + """
n7 = g.add_operator(graph.OperatorType.SAMPLE, [n6])
n8 = g.add_constant(1.0)
n9 = g.add_constant(tensor(-0.010050326585769653))
n10 = g.add_constant(tensor(-4.605170249938965))
n11 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n10])
n12 = g.add_operator(graph.OperatorType.ADD, [n9, n11])
n13 = g.add_constant(tensor(-4.605170249938965))
n14 = g.add_operator(graph.OperatorType.MULTIPLY, [n7, n13])
n15 = g.add_operator(graph.OperatorType.ADD, [n12, n14])
n16 = g.add_operator(graph.OperatorType.EXP, [n15])
n17 = g.add_operator(graph.OperatorType.NEGATE, [n16])
n18 = g.add_operator(graph.OperatorType.ADD, [n8, n17])
n19 = g.add_operator(graph.OperatorType.TO_REAL, [n18])
n20 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n19])"""  # noqa: B950
    + """
n21 = g.add_operator(graph.OperatorType.SAMPLE, [n20])
"""
)

expected_dot_1 = """
digraph "graph" {
  N0[label=0.009999999776482582];
  N10[label=-4.605170249938965];
  N11[label="*"];
  N12[label="+"];
  N13[label=-4.605170249938965];
  N14[label="*"];
  N15[label="+"];
  N16[label=Exp];
  N17[label="-"];
  N18[label="+"];
  N19[label=ToReal];
  N1[label=ToReal];
  N20[label=Bernoulli];
  N21[label=Sample];
  N2[label=Bernoulli];
  N3[label=Sample];
  N4[label=0.009999999776482582];
  N5[label=ToReal];
  N6[label=Bernoulli];
  N7[label=Sample];
  N8[label=1];
  N9[label=-0.010050326585769653];
  N1 -> N0[label=operand];
  N11 -> N10[label=right];
  N11 -> N3[label=left];
  N12 -> N11[label=right];
  N12 -> N9[label=left];
  N14 -> N13[label=right];
  N14 -> N7[label=left];
  N15 -> N12[label=left];
  N15 -> N14[label=right];
  N16 -> N15[label=operand];
  N17 -> N16[label=operand];
  N18 -> N17[label=right];
  N18 -> N8[label=left];
  N19 -> N18[label=operand];
  N2 -> N1[label=probability];
  N20 -> N19[label=probability];
  N21 -> N20[label=operand];
  N3 -> N2[label=operand];
  N5 -> N4[label=operand];
  N6 -> N5[label=probability];
  N7 -> N6[label=operand];
}
"""

expected_cpp_1 = """
graph::Graph g;
uint n0 = g.add_constant(torch::from_blob((float[]){0.009999999776482582}, {}));
uint n1 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n0}));
uint n2 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n4 = g.add_constant(torch::from_blob((float[]){0.009999999776482582}, {}));
uint n5 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n4}));
uint n6 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n5}));
uint n7 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n6}));
uint n8 = g.add_constant(1.0);
uint n9 = g.add_constant(torch::from_blob((float[]){-0.010050326585769653}, {}));
uint n10 = g.add_constant(torch::from_blob((float[]){-4.605170249938965}, {}));
uint n11 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n3, n10}));
uint n12 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n9, n11}));
uint n13 = g.add_constant(torch::from_blob((float[]){-4.605170249938965}, {}));
uint n14 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n7, n13}));
uint n15 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n12, n14}));
uint n16 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n15}));
uint n17 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n16}));
uint n18 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n8, n17}));
uint n19 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n18}));
uint n20 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n19}));
uint n21 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n20}));
"""

expected_bmg_1 = """
Node 0 type 1 parents [ ] children [ 1 ] tensor value 0.01
[ CPUFloatType{} ]
Node 1 type 3 parents [ 0 ] children [ 2 ] unknown value
Node 2 type 2 parents [ 1 ] children [ 3 ] unknown value
Node 3 type 3 parents [ 2 ] children [ 11 ] unknown value
Node 4 type 1 parents [ ] children [ 5 ] tensor value 0.01
[ CPUFloatType{} ]
Node 5 type 3 parents [ 4 ] children [ 6 ] unknown value
Node 6 type 2 parents [ 5 ] children [ 7 ] unknown value
Node 7 type 3 parents [ 6 ] children [ 14 ] unknown value
Node 8 type 1 parents [ ] children [ 18 ] real value 1
Node 9 type 1 parents [ ] children [ 12 ] tensor value -0.0100503
[ CPUFloatType{} ]
Node 10 type 1 parents [ ] children [ 11 ] tensor value -4.60517
[ CPUFloatType{} ]
Node 11 type 3 parents [ 3 10 ] children [ 12 ] unknown value
Node 12 type 3 parents [ 9 11 ] children [ 15 ] unknown value
Node 13 type 1 parents [ ] children [ 14 ] tensor value -4.60517
[ CPUFloatType{} ]
Node 14 type 3 parents [ 7 13 ] children [ 15 ] unknown value
Node 15 type 3 parents [ 12 14 ] children [ 16 ] unknown value
Node 16 type 3 parents [ 15 ] children [ 17 ] unknown value
Node 17 type 3 parents [ 16 ] children [ 18 ] unknown value
Node 18 type 3 parents [ 8 17 ] children [ 19 ] unknown value
Node 19 type 3 parents [ 18 ] children [ 20 ] unknown value
Node 20 type 2 parents [ 19 ] children [ 21 ] unknown value
Node 21 type 3 parents [ 20 ] children [ ] unknown value
"""


class CompilerTest(unittest.TestCase):
    def test_to_python_raw(self) -> None:
        """Tests for to_python_raw from bm_to_bmg.py"""
        self.maxDiff = None
        observed = to_python_raw(source1)
        self.assertEqual(observed.strip(), expected_raw_1.strip())

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
