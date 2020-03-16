# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_graph_builder.py"""
import unittest

from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from torch import tensor


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


class BMGraphBuilderTest(unittest.TestCase):
    def test_1(self) -> None:
        """Test 1"""

        t = tensor([[10, 20], [40, 50]])
        bmg = BMGraphBuilder()
        half = bmg.add_real(0.5)
        two = bmg.add_real(2)
        tens = bmg.add_tensor(t)
        tr = bmg.add_boolean(True)
        flip = bmg.add_bernoulli(half)
        samp = bmg.add_sample(flip)
        real = bmg.add_to_real(samp)
        neg = bmg.add_negate(real)
        add = bmg.add_addition(two, neg)
        add_t = bmg.add_to_tensor(add)
        bmg.add_multiplication(tens, add_t)
        bmg.add_observation(samp, tr)

        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=0.5];
  N10[label="*"];
  N11[label=Observation];
  N1[label=2];
  N2[label="[[10,20],\\\\n[40,50]]"];
  N3[label=True];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label=ToReal];
  N7[label="-"];
  N8[label="+"];
  N9[label=ToTensor];
  N10 -> N2[label=left];
  N10 -> N9[label=right];
  N11 -> N3[label=value];
  N11 -> N5[label=operand];
  N4 -> N0[label=probability];
  N5 -> N4[label=operand];
  N6 -> N5[label=operand];
  N7 -> N6[label=operand];
  N8 -> N1[label=left];
  N8 -> N7[label=right];
  N9 -> N8[label=operand];
}
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

        g = bmg.to_bmg()
        observed = g.to_string()
        expected = """
Node 0 type 1 parents [ ] children [ 4 ] real value 0.5
Node 1 type 1 parents [ ] children [ 8 ] real value 2
Node 2 type 1 parents [ ] children [ 9 ] tensor value  10  20
 40  50
[ CPULongType{2,2} ]
Node 3 type 1 parents [ ] children [ ] boolean value 1
Node 4 type 2 parents [ 0 ] children [ 5 ] unknown value
Node 5 type 3 parents [ 4 ] children [ 6 ] boolean value 1
Node 6 type 3 parents [ 5 ] children [ 7 ] unknown value
Node 7 type 3 parents [ 6 ] children [ 8 ] unknown value
Node 8 type 3 parents [ 1 7 ] children [ 9 ] unknown value
Node 9 type 3 parents [ 8 ] children [ 10 ] unknown value
Node 10 type 3 parents [ 2 9 ] children [ ] unknown value
        """
        # TODO: This test is disabled due to problems with how torch dumps out tensors
        # TODO: self.assertEqual(tidy(observed), tidy(expected))

        observed = bmg.to_python()

        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(0.5)
n1 = g.add_constant(2.0)
n2 = g.add_constant(tensor([[10,20],[40,50]]))
n3 = g.add_constant(True)
n4 = g.add_distribution(graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n0])
n5 = g.add_operator(graph.OperatorType.SAMPLE, [n4])
n6 = g.add_operator(graph.OperatorType.TO_REAL, [n5])
n7 = g.add_operator(graph.OperatorType.NEGATE, [n6])
n8 = g.add_operator(graph.OperatorType.ADD, [n1, n7])
n9 = g.add_operator(graph.OperatorType.TO_TENSOR, [n8])
n10 = g.add_operator(graph.OperatorType.MULTIPLY, [n2, n9])
g.observe(n5, True)
"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = bmg.to_cpp()

        expected = """
graph::Graph g;
uint n0 = g.add_constant(0.5);
uint n1 = g.add_constant(2.0);
uint n2 = g.add_constant(torch::from_blob((float[]){10,20,40,50}, {2,2}));
uint n3 = g.add_constant(true);
uint n4 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n4}));
uint n6 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n5}));
uint n7 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n1, n7}));
uint n9 = g.add_operator(
  graph::OperatorType::TO_TENSOR, std::vector<uint>({n8}));
uint n10 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n2, n9}));
g.observe([n5], true);
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_2(self) -> None:
        """Test 2"""

        # TODO: We haven't implemented DIVISION or POWER or LOG or NOT in BMG
        # TODO: C++ code yet.
        # TODO: When we do, update this test to show that we're representing
        # TODO: it correctly.

        bmg = BMGraphBuilder()
        one = bmg.add_real(1)
        two = bmg.add_real(2)
        # This should be folded:
        four = bmg.add_power(two, two)
        half = bmg.add_division(one, four)
        flip = bmg.add_bernoulli(half)
        samp = bmg.add_sample(flip)
        inv = bmg.add_not(samp)
        real = bmg.add_to_real(inv)
        div = bmg.add_division(real, two)
        pow = bmg.add_power(div, two)
        lg = bmg.add_log(pow)
        bmg.remove_orphans([lg])
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=0.25];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=not];
  N4[label=ToReal];
  N5[label=2];
  N6[label="/"];
  N7[label="**"];
  N8[label=Log];
  N1 -> N0[label=probability];
  N2 -> N1[label=operand];
  N3 -> N2[label=operand];
  N4 -> N3[label=operand];
  N6 -> N4[label=left];
  N6 -> N5[label=right];
  N7 -> N5[label=right];
  N7 -> N6[label=left];
  N8 -> N7[label=operand];
}
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())
