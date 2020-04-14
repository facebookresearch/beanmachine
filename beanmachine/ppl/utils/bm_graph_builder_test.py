# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_graph_builder.py"""
import math
import unittest
from typing import Any

import torch
from beanmachine.ppl.utils.bm_graph_builder import (
    AdditionNode,
    BernoulliNode,
    BMGraphBuilder,
    BooleanNode,
    DivisionNode,
    ExpNode,
    LogNode,
    MatrixMultiplicationNode,
    MultiplicationNode,
    NegateNode,
    NotNode,
    PowerNode,
    RealNode,
    SampleNode,
    SetOfTensors,
    TensorNode,
    ToRealNode,
)
from torch import Size, Tensor, tensor
from torch.distributions import Bernoulli


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


def tensor_equality(x: Tensor, y: Tensor) -> bool:
    # Tensor equality is weird.  Suppose x and y are both
    # tensor([1.0, 2.0]). Then x.eq(y) is tensor([True, True]),
    # and x.eq(y).all() is tensor(True).
    return bool(x.eq(y).all())


class BMGraphBuilderTest(unittest.TestCase):
    def assertEqual(self, x: Any, y: Any) -> bool:
        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return tensor_equality(x, y)
        return super().assertEqual(x, y)

    def test_1(self) -> None:
        """Test 1"""

        t = tensor([[10, 20], [40, 50]])
        bmg = BMGraphBuilder()
        half = bmg.add_probability(0.5)
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
n0 = g.add_constant_probability(0.5)
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
uint n0 = g.add_constant_probability(0.5);
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
        # Note that the orphan node "1" is not stripped out.
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
        bmg.add_log(pow)
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=1];
  N10[label=Log];
  N1[label=2];
  N2[label=4];
  N3[label=0.25];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label=not];
  N7[label=ToReal];
  N8[label="/"];
  N9[label="**"];
  N10 -> N9[label=operand];
  N4 -> N3[label=probability];
  N5 -> N4[label=operand];
  N6 -> N5[label=operand];
  N7 -> N6[label=operand];
  N8 -> N1[label=right];
  N8 -> N7[label=left];
  N9 -> N1[label=right];
  N9 -> N8[label=left];
}
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

    def test_3(self) -> None:
        """Test 3"""
        bmg = BMGraphBuilder()
        self.assertTrue(bmg.add_real(1.0))
        self.assertTrue(bmg.add_boolean(True))
        self.assertTrue(bmg.add_tensor(tensor(True)))
        self.assertTrue(bmg.add_tensor(tensor(1.0)))
        self.assertTrue(bmg.add_tensor(tensor([1.0])))
        self.assertFalse(bmg.add_real(0.0))
        self.assertFalse(bmg.add_boolean(False))
        self.assertFalse(bmg.add_tensor(tensor(False)))
        self.assertFalse(bmg.add_tensor(tensor(0.0)))
        self.assertFalse(bmg.add_tensor(tensor([0.0])))

    # The "add" methods do exactly that: add a node to the graph if it is not
    # already there.
    #
    # The "handle" methods try to keep everything in unwrapped values if possible;
    # they are trying to keep values out of the graph when possible.
    #
    # The next few tests verify that the handle functions are working as designed.

    def test_handle_bernoulli(self) -> None:
        # This test verifies that various mechanisms for producing an addition node
        # in the graph are working as designed.

        # TODO: Test tensors also.

        bmg = BMGraphBuilder()

        # TODO: Should handle_bernoulli given constants just produce a Bernoulli object
        # TODO: as a value?
        # TODO: Do we actually need handle_bernoulli at all? It seems like we could
        # TODO: simply delete it and use the logic that is in handle_sample.
        b = bmg.handle_bernoulli(0.5)
        self.assertTrue(isinstance(b, BernoulliNode))

        r = bmg.add_real(0.5)
        b = bmg.handle_bernoulli(r)
        self.assertTrue(isinstance(b, BernoulliNode))

    def test_handle_sample(self) -> None:

        bmg = BMGraphBuilder()

        # Sample on a graph node.
        b = bmg.add_bernoulli(bmg.add_tensor(tensor(0.5)))
        s1 = bmg.handle_sample(b)
        self.assertTrue(isinstance(s1, SampleNode))

        # Sample on a distribution object.
        b = Bernoulli(0.5)
        s2 = bmg.handle_sample(b)
        self.assertTrue(isinstance(s2, SampleNode))

        # Verify that they are not memoized; samples are always distinct.
        self.assertFalse(s1 is s2)

    def test_addition(self) -> None:
        """Test addition"""

        # This test verifies that various mechanisms for producing an addition node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor(1.0)
        t2 = tensor(2.0)
        t3 = tensor(3.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch defines a "static" add method that takes two values.
        # Calling torch.add(x, y) should be logically the same as x + y

        ta = torch.add
        self.assertEqual(bmg.handle_dot_get(torch, "add"), ta)

        # torch defines an "instance" add method that takes a value.
        # Calling Tensor.add(x, y) or x.add(y) should be logically the same as x + y.

        # TODO: In Tensor.add(x, y), x is required to be a tensor, not a double. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0, 2.0]) would not fail.
        # TODO: Should it?

        ta1 = t1.add
        self.assertEqual(bmg.handle_dot_get(t1, "add"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "add")

        ta2 = torch.Tensor.add
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "add"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "add")

        # Adding two values produces a value
        self.assertEqual(bmg.handle_addition(1.0, 2.0), 3.0)
        self.assertEqual(bmg.handle_addition(1.0, t2), t3)
        self.assertEqual(bmg.handle_addition(t1, 2.0), t3)
        self.assertEqual(bmg.handle_addition(t1, t2), t3)
        self.assertEqual(bmg.handle_function(ta, [1.0], {"other": 2.0}), t3)
        self.assertEqual(bmg.handle_function(ta, [1.0, t2]), t3)
        self.assertEqual(bmg.handle_function(ta, [t1, 2.0]), t3)
        self.assertEqual(bmg.handle_function(ta, [t1, t2]), t3)
        self.assertEqual(bmg.handle_function(ta1, [2.0]), t3)
        self.assertEqual(bmg.handle_function(ta1, [], {"other": 2.0}), t3)
        self.assertEqual(bmg.handle_function(ta1, [t2]), t3)
        self.assertEqual(bmg.handle_function(ta2, [t1, 2.0]), t3)
        self.assertEqual(bmg.handle_function(ta2, [t1, t2]), t3)
        self.assertEqual(bmg.handle_function(ta2, [t1], {"other": t2}), t3)

        # Adding a graph constant and a value produces a value
        self.assertEqual(bmg.handle_addition(gr1, 2.0), 3.0)
        self.assertEqual(bmg.handle_addition(gr1, t2), t3)
        self.assertEqual(bmg.handle_addition(gt1, 2.0), t3)
        self.assertEqual(bmg.handle_addition(gt1, t2), t3)
        self.assertEqual(bmg.handle_addition(2.0, gr1), 3.0)
        self.assertEqual(bmg.handle_addition(2.0, gt1), t3)
        self.assertEqual(bmg.handle_addition(t2, gr1), t3)
        self.assertEqual(bmg.handle_addition(t2, gt1), t3)
        self.assertEqual(bmg.handle_function(ta, [gr1, 2.0]), t3)
        self.assertEqual(bmg.handle_function(ta, [gr1, t2]), t3)
        self.assertEqual(bmg.handle_function(ta, [gt1, 2.0]), t3)
        self.assertEqual(bmg.handle_function(ta, [gt1, t2]), t3)
        self.assertEqual(bmg.handle_function(gta1, [2.0]), t3)
        self.assertEqual(bmg.handle_function(gta1, [t2]), t3)
        self.assertEqual(bmg.handle_function(ta2, [gt1, 2.0]), t3)
        self.assertEqual(bmg.handle_function(ta2, [gt1, t2]), t3)
        self.assertEqual(bmg.handle_function(ta, [2.0, gr1]), t3)
        self.assertEqual(bmg.handle_function(ta, [2.0, gt1]), t3)
        self.assertEqual(bmg.handle_function(ta, [t2, gr1]), t3)
        self.assertEqual(bmg.handle_function(ta, [t2, gt1]), t3)
        self.assertEqual(bmg.handle_function(ta1, [gr1]), t2)
        self.assertEqual(bmg.handle_function(ta1, [gt1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t2, gr1]), t3)
        self.assertEqual(bmg.handle_function(ta2, [t2, gt1]), t3)

        # Adding two graph constants produces a value
        self.assertEqual(bmg.handle_addition(gr1, gr1), 2.0)
        self.assertEqual(bmg.handle_addition(gr1, gt1), t2)
        self.assertEqual(bmg.handle_addition(gt1, gr1), t2)
        self.assertEqual(bmg.handle_addition(gt1, gt1), t2)
        self.assertEqual(bmg.handle_function(ta, [gr1, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gr1, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1, gt1]), t2)
        self.assertEqual(bmg.handle_function(gta1, [gr1]), t2)
        self.assertEqual(bmg.handle_function(gta1, [gt1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gt1]), t2)

        # Sample plus value produces node
        n = AdditionNode
        self.assertTrue(isinstance(bmg.handle_addition(s, 2.0), n))
        self.assertTrue(isinstance(bmg.handle_addition(s, t2), n))
        self.assertTrue(isinstance(bmg.handle_addition(2.0, s), n))
        self.assertTrue(isinstance(bmg.handle_addition(t2, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [2.0, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [t2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2, s]), n))

        # Sample plus graph node produces node
        self.assertTrue(isinstance(bmg.handle_addition(s, gr1), n))
        self.assertTrue(isinstance(bmg.handle_addition(s, gt1), n))
        self.assertTrue(isinstance(bmg.handle_addition(gr1, s), n))
        self.assertTrue(isinstance(bmg.handle_addition(gt1, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gr1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1, s]), n))

    def test_division(self) -> None:
        """Test division"""

        # This test verifies that various mechanisms for producing a division node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor(1.0)
        t2 = tensor(2.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gr2 = bmg.add_real(2.0)
        self.assertTrue(isinstance(gr2, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))
        gt2 = bmg.add_tensor(t2)
        self.assertTrue(isinstance(gt2, TensorNode))

        # torch defines a "static" div method that takes two values.
        # Calling torch.div(x, y) should be logically the same as x + y

        ta = torch.div
        self.assertEqual(bmg.handle_dot_get(torch, "div"), ta)

        # torch defines an "instance" div method that takes a value.
        # Calling Tensor.div(x, y) or x.div(y) should be logically the same as x + y.

        # TODO: In Tensor.div(x, y), x is required to be a tensor, not a double. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0, 2.0]) would not fail.
        # TODO: Should it?

        ta1 = t2.div
        self.assertEqual(bmg.handle_dot_get(t2, "div"), ta1)

        gta1 = bmg.handle_dot_get(gt2, "div")

        ta2 = torch.Tensor.div
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "div"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "div")

        # Dividing two values produces a value
        self.assertEqual(bmg.handle_division(2.0, 1.0), 2.0)
        self.assertEqual(bmg.handle_division(2.0, t1), t2)
        self.assertEqual(bmg.handle_division(t2, 1.0), t2)
        self.assertEqual(bmg.handle_division(t2, t1), t2)
        self.assertEqual(bmg.handle_function(ta, [2.0], {"other": 1.0}), t2)
        self.assertEqual(bmg.handle_function(ta, [2.0, t1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t2, 1.0]), t2)
        self.assertEqual(bmg.handle_function(ta, [t2, t1]), t2)
        self.assertEqual(bmg.handle_function(ta1, [1.0]), t2)
        self.assertEqual(bmg.handle_function(ta1, [], {"other": 1.0}), t2)
        self.assertEqual(bmg.handle_function(ta1, [t1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t2, 1.0]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t2], {"other": 1.0}), t2)
        self.assertEqual(bmg.handle_function(ta2, [t2, t1]), t2)

        # Dividing a graph constant and a value produces a value
        self.assertEqual(bmg.handle_division(gr2, 2.0), 1.0)
        self.assertEqual(bmg.handle_division(gr2, t2), t1)
        self.assertEqual(bmg.handle_division(gt2, 2.0), t1)
        self.assertEqual(bmg.handle_division(gt2, t2), t1)
        self.assertEqual(bmg.handle_division(2.0, gr2), 1.0)
        self.assertEqual(bmg.handle_division(2.0, gt2), t1)
        self.assertEqual(bmg.handle_division(t2, gr2), t1)
        self.assertEqual(bmg.handle_division(t2, gt2), t1)
        self.assertEqual(bmg.handle_function(ta, [gr2, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta, [gr2, t2]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt2, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt2, t2]), t1)
        self.assertEqual(bmg.handle_function(gta1, [2.0]), t1)
        self.assertEqual(bmg.handle_function(gta1, [t2]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt2, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt2, t2]), t1)
        self.assertEqual(bmg.handle_function(ta, [2.0, gr2]), t1)
        self.assertEqual(bmg.handle_function(ta, [2.0], {"other": gr2}), t1)
        self.assertEqual(bmg.handle_function(ta, [2.0, gt2]), t1)
        self.assertEqual(bmg.handle_function(ta, [t2, gr2]), t1)
        self.assertEqual(bmg.handle_function(ta, [t2, gt2]), t1)
        self.assertEqual(bmg.handle_function(ta1, [gr2]), t1)
        self.assertEqual(bmg.handle_function(ta1, [gt2]), t1)
        self.assertEqual(bmg.handle_function(ta2, [t2, gr2]), t1)
        self.assertEqual(bmg.handle_function(ta2, [t2, gt2]), t1)

        # Dividing two graph constants produces a value
        self.assertEqual(bmg.handle_division(gr2, gr1), 2.0)
        self.assertEqual(bmg.handle_division(gr2, gt1), t2)
        self.assertEqual(bmg.handle_division(gt2, gr1), t2)
        self.assertEqual(bmg.handle_division(gt2, gt1), t2)
        self.assertEqual(bmg.handle_function(ta, [gr2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gr2, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt2, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt2], {"other": gt1}), t2)
        self.assertEqual(bmg.handle_function(gta1, [gr1]), t2)
        self.assertEqual(bmg.handle_function(gta1, [gt1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt2, gt1]), t2)

        # Sample divided by value produces node
        n = DivisionNode
        self.assertTrue(isinstance(bmg.handle_division(s, 2.0), n))
        self.assertTrue(isinstance(bmg.handle_division(s, t2), n))
        self.assertTrue(isinstance(bmg.handle_division(2.0, s), n))
        self.assertTrue(isinstance(bmg.handle_division(t2, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [2.0, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [t2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [], {"other": s}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2, s]), n))

        # Sample divided by graph node produces node
        self.assertTrue(isinstance(bmg.handle_division(s, gr1), n))
        self.assertTrue(isinstance(bmg.handle_division(s, gt1), n))
        self.assertTrue(isinstance(bmg.handle_division(gr1, s), n))
        self.assertTrue(isinstance(bmg.handle_division(gt1, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gr1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1, s]), n))

    def test_exp(self) -> None:
        """Test exp"""

        # This test verifies that various mechanisms for producing an exp node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        e = math.exp(1.0)
        t1 = tensor(1.0)
        te = torch.exp(t1)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch defines a "static" exp method that takes one value.
        # TODO: torch.exp(x) requires that x be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta, [1.0]) would not fail.
        # TODO: Should it?

        ta = torch.exp
        self.assertEqual(bmg.handle_dot_get(torch, "exp"), ta)

        # torch defines an "instance" exp method that takes no arguments.
        # Calling Tensor.exp(x) or x.exp() should produce an exp node.

        # TODO: In Tensor.exp(x), x is required to be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0]) would not fail.
        # TODO: Should it?

        ta1 = t1.exp
        self.assertEqual(bmg.handle_dot_get(t1, "exp"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "exp")

        ta2 = torch.Tensor.exp
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "exp"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "exp")

        # Exp of a value produces a value
        self.assertEqual(bmg.handle_exp(1.0), e)
        self.assertEqual(bmg.handle_exp(t1), te)
        # Not legal: self.assertEqual(bmg.handle_function(ta, [1.0]), e)
        self.assertEqual(bmg.handle_function(ta, [t1]), te)
        self.assertEqual(bmg.handle_function(ta1, []), te)
        self.assertEqual(bmg.handle_function(ta2, [t1]), te)

        # Exp of a graph constant produces a value
        self.assertEqual(bmg.handle_exp(gr1), e)
        self.assertEqual(bmg.handle_exp(gt1), te)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta, [gr1]), e)
        self.assertEqual(bmg.handle_function(ta, [gt1]), te)
        self.assertEqual(bmg.handle_function(gta1, []), te)

        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta2, [gr1]), e)
        self.assertEqual(bmg.handle_function(ta2, [gt1]), te)

        # Exp of sample produces node
        n = ExpNode
        self.assertTrue(isinstance(bmg.handle_exp(s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_log(self) -> None:
        """Test log"""

        # This test verifies that various mechanisms for producing an exp node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t0 = tensor(0.0)
        t1 = tensor(1.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch defines a "static" log method that takes one value.
        # TODO: torch.log(x) requires that x be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta, [1.0]) would not fail.
        # TODO: Should it?

        ta = torch.log
        self.assertEqual(bmg.handle_dot_get(torch, "log"), ta)

        # torch defines an "instance" log method that takes no arguments.
        # Calling Tensor.log(x) or x.log() should produce a log node.

        # TODO: In Tensor.log(x), x is required to be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0]) would not fail.
        # TODO: Should it?

        ta1 = t1.log
        self.assertEqual(bmg.handle_dot_get(t1, "log"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "log")

        ta2 = torch.Tensor.log
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "log"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "log")

        # Log of a value produces a value
        self.assertEqual(bmg.handle_log(1.0), 0.0)
        self.assertEqual(bmg.handle_log(t1), t0)
        # Not legal: self.assertEqual(bmg.handle_function(ta, [1.0]), 0.0)
        self.assertEqual(bmg.handle_function(ta, [t1]), t0)
        self.assertEqual(bmg.handle_function(ta1, []), t0)
        self.assertEqual(bmg.handle_function(ta2, [t1]), t0)

        # Log of a graph constant produces a value
        self.assertEqual(bmg.handle_log(gr1), 0.0)
        self.assertEqual(bmg.handle_log(gt1), t0)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta, [gr1]), 0.0)
        self.assertEqual(bmg.handle_function(ta, [gt1]), t0)
        self.assertEqual(bmg.handle_function(gta1, []), t0)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta2, [gr1]), 0.0)
        self.assertEqual(bmg.handle_function(ta2, [gt1]), t0)

        # Log of sample produces node
        n = LogNode
        self.assertTrue(isinstance(bmg.handle_log(s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_multiplication(self) -> None:
        """Test multiplication"""

        # This test verifies that various mechanisms for producing a multiplication node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor(1.0)
        t2 = tensor(2.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch defines a "static" mul method that takes two values.
        # Calling torch.mul(x, y) should be logically the same as x * y

        ta = torch.mul
        self.assertEqual(bmg.handle_dot_get(torch, "mul"), ta)

        # torch defines an "instance" mul method that takes a value.
        # Calling Tensor.mul(x, y) or x.mul(y) should be logically the same as x * y.

        # TODO: In Tensor.mul(x, y), x is required to be a tensor, not a double. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0, 2.0]) would not fail.
        # TODO: Should it?

        ta1 = t1.mul
        self.assertEqual(bmg.handle_dot_get(t1, "mul"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "mul")

        ta2 = torch.Tensor.mul
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "mul"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "mul")

        # Multiplying two values produces a value
        self.assertEqual(bmg.handle_multiplication(1.0, 2.0), 2.0)
        self.assertEqual(bmg.handle_multiplication(1.0, t2), t2)
        self.assertEqual(bmg.handle_multiplication(t1, 2.0), t2)
        self.assertEqual(bmg.handle_multiplication(t1, t2), t2)
        self.assertEqual(bmg.handle_function(ta, [1.0], {"other": 2.0}), t2)
        self.assertEqual(bmg.handle_function(ta, [1.0, t2]), t2)
        self.assertEqual(bmg.handle_function(ta, [t1, 2.0]), t2)
        self.assertEqual(bmg.handle_function(ta, [t1, t2]), t2)
        self.assertEqual(bmg.handle_function(ta1, [2.0]), t2)
        self.assertEqual(bmg.handle_function(ta1, [], {"other": 2.0}), t2)
        self.assertEqual(bmg.handle_function(ta1, [t2]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1, 2.0]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1], {"other": 2.0}), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1, t2]), t2)

        # Multiplying a graph constant and a value produces a value
        self.assertEqual(bmg.handle_multiplication(gr1, 2.0), 2.0)
        self.assertEqual(bmg.handle_multiplication(gr1, t2), t2)
        self.assertEqual(bmg.handle_multiplication(gt1, 2.0), t2)
        self.assertEqual(bmg.handle_multiplication(gt1, t2), t2)
        self.assertEqual(bmg.handle_multiplication(2.0, gr1), 2.0)
        self.assertEqual(bmg.handle_multiplication(2.0, gt1), t2)
        self.assertEqual(bmg.handle_multiplication(t2, gr1), t2)
        self.assertEqual(bmg.handle_multiplication(t2, gt1), t2)
        self.assertEqual(bmg.handle_function(ta, [gr1, 2.0]), t2)
        self.assertEqual(bmg.handle_function(ta, [gr1, t2]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1, 2.0]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1, t2]), t2)
        self.assertEqual(bmg.handle_function(gta1, [2.0]), t2)
        self.assertEqual(bmg.handle_function(gta1, [t2]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt1, 2.0]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt1, t2]), t2)
        self.assertEqual(bmg.handle_function(ta, [2.0, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [2.0, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t2, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta1, [gr1]), t1)
        self.assertEqual(bmg.handle_function(ta1, [gt1]), t1)
        self.assertEqual(bmg.handle_function(ta1, [], {"other": gt1}), t1)
        self.assertEqual(bmg.handle_function(ta2, [t2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t2, gt1]), t2)

        # Multiplying two graph constants produces a value
        self.assertEqual(bmg.handle_multiplication(gr1, gr1), 1.0)
        self.assertEqual(bmg.handle_multiplication(gr1, gt1), t1)
        self.assertEqual(bmg.handle_multiplication(gt1, gr1), t1)
        self.assertEqual(bmg.handle_multiplication(gt1, gt1), t1)
        self.assertEqual(bmg.handle_function(ta, [gr1, gr1]), t1)
        self.assertEqual(bmg.handle_function(ta, [gr1, gt1]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt1, gr1]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt1, gt1]), t1)
        self.assertEqual(bmg.handle_function(gta1, [gr1]), t1)
        self.assertEqual(bmg.handle_function(gta1, [gt1]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gr1]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gt1]), t1)

        # Sample times value produces node
        n = MultiplicationNode
        self.assertTrue(isinstance(bmg.handle_multiplication(s, 2.0), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(s, t2), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(2.0, s), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(t2, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [2.0, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [t2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2, s]), n))

        # Sample times graph node produces node
        self.assertTrue(isinstance(bmg.handle_multiplication(s, gr1), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(s, gt1), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(gr1, s), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(gt1, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gr1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1], {"other": s}), n))

    def test_matrix_multiplication(self) -> None:
        """Test matrix_multiplication"""

        # This test verifies that various mechanisms for producing a matrix
        # multiplication node in the graph -- or avoiding producing such a
        # node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor([[3.0, 4.0], [5.0, 6.0]])
        t2 = tensor([[29.0, 36.0], [45.0, 56.0]])

        # Graph nodes
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))
        gt2 = bmg.add_tensor(t2)
        self.assertTrue(isinstance(gt2, TensorNode))

        # torch defines a "static" mm method that takes two values.

        ta = torch.mm
        self.assertEqual(bmg.handle_dot_get(torch, "mm"), ta)

        # torch defines an "instance" mm method that takes a value.

        # TODO: In Tensor.mm(x, y), x and y are required to be a tensor, not a double.
        # TODO: Consider enforcing this rule.

        ta1 = t1.mm
        self.assertEqual(bmg.handle_dot_get(t1, "mm"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "mm")

        ta2 = torch.Tensor.mm
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "mm"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(
            bmg.add_bernoulli(bmg.add_tensor(tensor([[0.5, 0.5], [0.5, 0.5]])))
        )
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "mm")

        # Multiplying two values produces a value
        self.assertEqual(bmg.handle_matrix_multiplication(t1, t1), t2)
        self.assertEqual(bmg.handle_function(ta, [t1, t1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t1], {"mat2": t1}), t2)
        self.assertEqual(bmg.handle_function(ta1, [t1]), t2)
        self.assertEqual(bmg.handle_function(ta1, [], {"mat2": t1}), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1, t1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1], {"mat2": t1}), t2)

        # Multiplying a graph constant and a value produces a value
        self.assertEqual(bmg.handle_matrix_multiplication(gt1, t1), t2)
        self.assertEqual(bmg.handle_matrix_multiplication(t1, gt1), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1, t1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1], {"mat2": t1}), t2)
        self.assertEqual(bmg.handle_function(gta1, [t1]), t2)
        self.assertEqual(bmg.handle_function(gta1, [], {"mat2": t1}), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt1, t1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [gt1], {"mat2": t1}), t2)
        self.assertEqual(bmg.handle_function(ta, [t1, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t1], {"mat2": gt1}), t2)
        self.assertEqual(bmg.handle_function(ta1, [gt1]), t2)
        self.assertEqual(bmg.handle_function(ta1, [], {"mat2": gt1}), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t1], {"mat2": gt1}), t2)

        # Multiplying two graph constants produces a value
        self.assertEqual(bmg.handle_matrix_multiplication(gt1, gt1), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [gt1], {"mat2": gt1}), t2)
        self.assertEqual(bmg.handle_function(gta1, [gt1]), t1)
        self.assertEqual(bmg.handle_function(gta1, [], {"mat2": gt1}), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gt1]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1], {"mat2": gt1}), t1)

        # Sample times value produces node
        n = MatrixMultiplicationNode
        self.assertTrue(isinstance(bmg.handle_matrix_multiplication(s, t1), n))
        self.assertTrue(isinstance(bmg.handle_matrix_multiplication(t1, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, t1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s], {"mat2": t1}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [t1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [t1], {"mat2": s}), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [t1]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [], {"mat2": t1}), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [], {"mat2": s}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, t1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s], {"mat2": t1}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2], {"mat2": s}), n))

        # Sample times graph node produces node
        self.assertTrue(isinstance(bmg.handle_matrix_multiplication(s, gt1), n))
        self.assertTrue(isinstance(bmg.handle_matrix_multiplication(gt1, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s], {"mat2": gt1}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt1], {"mat2": s}), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [], {"mat2": gt1}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s], {"mat2": gt1}), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1], {"mat2": s}), n))

    def test_negation(self) -> None:
        """Test negation"""

        # This test verifies that various mechanisms for producing a negation node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor(1.0)
        t2 = tensor(2.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch defines a "static" neg method that takes one value.
        # Calling torch.neg(x) should be logically the same as -x
        # TODO: torch.neg(x) requires that x be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta, [1.0]) would not fail.
        # TODO: Should it?

        ta = torch.neg
        self.assertEqual(bmg.handle_dot_get(torch, "neg"), ta)

        # torch defines an "instance" neg method that takes no arguments.
        # Calling Tensor.neg(x) or x.neg() should be logically the same as -x.

        # TODO: In Tensor.neg(x), x is required to be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0]) would not fail.
        # TODO: Should it?

        ta1 = t1.neg
        self.assertEqual(bmg.handle_dot_get(t1, "neg"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "neg")

        ta2 = torch.Tensor.neg
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "neg"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "neg")

        # Negating a value produces a value
        self.assertEqual(bmg.handle_negate(1.0), -1.0)
        self.assertEqual(bmg.handle_negate(t2), -t2)
        # Not legal: self.assertEqual(bmg.handle_function(ta, [1.0]), -1.0)
        self.assertEqual(bmg.handle_function(ta, [t2]), -t2)
        self.assertEqual(bmg.handle_function(ta1, []), -t1)
        self.assertEqual(bmg.handle_function(ta2, [t2]), -t2)

        # Negating a graph constant produces a value
        self.assertEqual(bmg.handle_negate(gr1), -1.0)
        self.assertEqual(bmg.handle_negate(gt1), -t1)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta, [gr1]), -1.0)
        self.assertEqual(bmg.handle_function(ta, [gt1]), -t1)
        self.assertEqual(bmg.handle_function(ta, [], {"input": gt1}), -t1)
        self.assertEqual(bmg.handle_function(gta1, []), -t1)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta2, [gr1]), -1.0)
        self.assertEqual(bmg.handle_function(ta2, [gt1]), -t1)

        # Negating sample produces node
        n = NegateNode
        self.assertTrue(isinstance(bmg.handle_negate(s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_not(self) -> None:
        """Test not"""

        # This test verifies that various mechanisms for producing a logical-not node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        tt = tensor(True)
        tf = tensor(False)

        # Graph nodes
        gbt = bmg.add_boolean(True)
        self.assertTrue(isinstance(gbt, BooleanNode))
        gtt = bmg.add_tensor(tt)
        self.assertTrue(isinstance(gtt, TensorNode))

        # torch defines a "static" logical_not method that takes one value.
        # Calling torch.logical_not(x) should be logically the same as "not x"
        # TODO: torch.logical_not(x) requires that x be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta, [1.0]) would not fail.
        # TODO: Should it?

        ta = torch.logical_not
        self.assertEqual(bmg.handle_dot_get(torch, "logical_not"), ta)

        # torch defines an "instance" add method that takes no arguments.
        # Calling Tensor.logical_not(x) or x.logical_not() should be logically
        # the same as "not x".

        # TODO: In Tensor.logical_not(x), x is required to be a tensor, not a float.
        # TODO: We do not enforce this rule; handle_function(ta2, [1.0]) would not fail.
        # TODO: Should it?

        ta1 = tt.logical_not
        self.assertEqual(bmg.handle_dot_get(tt, "logical_not"), ta1)

        gta1 = bmg.handle_dot_get(gtt, "logical_not")

        ta2 = torch.Tensor.logical_not
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "logical_not"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "logical_not")

        # Negating a value produces a value
        self.assertEqual(bmg.handle_not(True), False)
        self.assertEqual(bmg.handle_not(tt), tf)
        # Not legal: self.assertEqual(bmg.handle_function(ta, [True]), False)
        self.assertEqual(bmg.handle_function(ta, [tt]), tf)
        self.assertEqual(bmg.handle_function(ta1, []), tf)
        self.assertEqual(bmg.handle_function(ta2, [tt]), tf)

        # Negating a graph constant produces a value
        self.assertEqual(bmg.handle_not(gbt), False)
        self.assertEqual(bmg.handle_not(gtt), tf)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta, [gbt]), False)
        self.assertEqual(bmg.handle_function(ta, [gtt]), tf)
        self.assertEqual(bmg.handle_function(ta, [], {"input": gtt}), tf)
        self.assertEqual(bmg.handle_function(gta1, []), tf)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta2, [gbt]), False)
        self.assertEqual(bmg.handle_function(ta2, [gtt]), tf)

        # Negating sample produces node
        n = NotNode
        self.assertTrue(isinstance(bmg.handle_not(s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_power(self) -> None:
        """Test power"""

        # This test verifies that various mechanisms for producing a power node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor(1.0)
        t2 = tensor(2.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch defines a "static" pow method that takes two values.
        # Calling torch.pow(x, y) should be logically the same as x ** y

        ta = torch.pow
        self.assertEqual(bmg.handle_dot_get(torch, "pow"), ta)

        # torch defines an "instance" pow method that takes a value.
        # Calling Tensor.pow(x, y) or x.pow(y) should be logically the same as x * y.

        # Note that unlike add, div, mul, the pow function on tensors takes only:
        # (tensor, tensor)
        # (number, tensor)
        # (tensor, number)
        # whereas the others allow (number, number).
        # TODO: Should we enforce this rule when the arguments are, say, samples?

        ta1 = t1.pow
        self.assertEqual(bmg.handle_dot_get(t1, "pow"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "pow")

        ta2 = torch.Tensor.pow
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "pow"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "pow")

        # Power of two values produces a value
        self.assertEqual(bmg.handle_power(1.0, 2.0), 1.0)
        self.assertEqual(bmg.handle_power(1.0, t2), t1)
        self.assertEqual(bmg.handle_power(t1, 2.0), t1)
        self.assertEqual(bmg.handle_power(t1, t2), t1)
        # Not legal: self.assertEqual(bmg.handle_function(ta, [1.0, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta, [1.0], {"exponent": t2}), t1)
        self.assertEqual(bmg.handle_function(ta, [t1, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta, [t1, t2]), t1)
        self.assertEqual(bmg.handle_function(ta1, [2.0]), t1)
        self.assertEqual(bmg.handle_function(ta1, [t2]), t1)
        self.assertEqual(bmg.handle_function(ta1, [], {"exponent": t2}), t1)
        self.assertEqual(bmg.handle_function(ta2, [t1, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta2, [t1, t2]), t1)
        self.assertEqual(bmg.handle_function(ta2, [t1], {"exponent": t2}), t1)

        # Power of a graph constant and a value produces a value
        self.assertEqual(bmg.handle_power(gr1, 2.0), 1.0)
        self.assertEqual(bmg.handle_power(gr1, t2), t1)
        self.assertEqual(bmg.handle_power(gt1, 2.0), t1)
        self.assertEqual(bmg.handle_power(gt1, t2), t1)
        self.assertEqual(bmg.handle_power(2.0, gr1), 2.0)
        self.assertEqual(bmg.handle_power(2.0, gt1), t2)
        self.assertEqual(bmg.handle_power(t2, gr1), t2)
        self.assertEqual(bmg.handle_power(t2, gt1), t2)
        self.assertEqual(bmg.handle_function(ta, [gr1, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta, [gr1, t2]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt1, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt1, t2]), t1)
        self.assertEqual(bmg.handle_function(gta1, [2.0]), t1)
        self.assertEqual(bmg.handle_function(gta1, [t2]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, 2.0]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, t2]), t1)
        self.assertEqual(bmg.handle_function(ta, [2.0, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [2.0, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta, [t2, gt1]), t2)
        self.assertEqual(bmg.handle_function(ta1, [gr1]), t1)
        self.assertEqual(bmg.handle_function(ta1, [gt1]), t1)
        self.assertEqual(bmg.handle_function(ta2, [t2, gr1]), t2)
        self.assertEqual(bmg.handle_function(ta2, [t2, gt1]), t2)

        # Power of two graph constants produces a value
        self.assertEqual(bmg.handle_power(gr1, gr1), 1.0)
        self.assertEqual(bmg.handle_power(gr1, gt1), t1)
        self.assertEqual(bmg.handle_power(gt1, gr1), t1)
        self.assertEqual(bmg.handle_power(gt1, gt1), t1)
        self.assertEqual(bmg.handle_function(ta, [gr1, gr1]), t1)
        self.assertEqual(bmg.handle_function(ta, [gr1, gt1]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt1, gr1]), t1)
        self.assertEqual(bmg.handle_function(ta, [gt1, gt1]), t1)
        self.assertEqual(bmg.handle_function(gta1, [gr1]), t1)
        self.assertEqual(bmg.handle_function(gta1, [gt1]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gr1]), t1)
        self.assertEqual(bmg.handle_function(ta2, [gt1, gt1]), t1)

        # Power of sample and value produces node
        n = PowerNode
        self.assertTrue(isinstance(bmg.handle_power(s, 2.0), n))
        self.assertTrue(isinstance(bmg.handle_power(s, t2), n))
        self.assertTrue(isinstance(bmg.handle_power(2.0, s), n))
        self.assertTrue(isinstance(bmg.handle_power(t2, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [2.0, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [t2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(gta1, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2, s]), n))

        # Power of sample and graph node produces node
        self.assertTrue(isinstance(bmg.handle_power(s, gr1), n))
        self.assertTrue(isinstance(bmg.handle_power(s, gt1), n))
        self.assertTrue(isinstance(bmg.handle_power(gr1, s), n))
        self.assertTrue(isinstance(bmg.handle_power(gt1, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gr1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt1, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gr1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gt1]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt1, s]), n))

    def test_to_real(self) -> None:
        """Test to_real"""

        # This test verifies that various mechanisms for producing a to_real node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor(1.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_tensor(t1)
        self.assertTrue(isinstance(gt1, TensorNode))

        # torch.float is not a function, unlike torch.log, torch.add and so on.

        # torch defines an "instance" float method that takes no arguments.
        # Calling Tensor.float(x) or x.float() should produce a to_real node.

        # TODO: In Tensor.float(x), x is required to be a tensor, not a float. We do
        # TODO: not enforce this rule; handle_function(ta2, [1.0]) would not fail.
        # TODO: Should it?

        ta1 = t1.float
        self.assertEqual(bmg.handle_dot_get(t1, "float"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "float")

        ta2 = torch.Tensor.float
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "float"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "float")

        # Float of a value produces a value
        self.assertEqual(bmg.handle_to_real(1.0), 1.0)
        self.assertEqual(bmg.handle_to_real(t1), 1.0)
        self.assertEqual(bmg.handle_function(ta1, []), 1.0)
        self.assertEqual(bmg.handle_function(ta2, [t1]), 1.0)

        # Float of a graph constant produces a value
        self.assertEqual(bmg.handle_to_real(gr1), 1.0)
        self.assertEqual(bmg.handle_to_real(gt1), 1.0)
        self.assertEqual(bmg.handle_function(gta1, []), 1.0)
        # TODO: Should this be illegal?
        self.assertEqual(bmg.handle_function(ta2, [gr1]), 1.0)
        self.assertEqual(bmg.handle_function(ta2, [gt1]), 1.0)

        # Float of sample produces node
        n = ToRealNode
        self.assertTrue(isinstance(bmg.handle_to_real(s), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_types(self) -> None:
        bmg = BMGraphBuilder()
        t = bmg.add_tensor(tensor(1.0))
        r = bmg.add_real(1.0)
        b = bmg.add_boolean(True)
        bern = bmg.add_bernoulli(t)
        s = bmg.add_sample(bern)
        # TODO: What should we do for things like bool plus real, and so on?
        # TODO: Are these errors, or do we introduce to_real nodes?
        self.assertEqual(t.node_type(), Tensor)
        self.assertEqual(r.node_type(), float)
        self.assertEqual(b.node_type(), bool)
        self.assertEqual(bern.node_type(), Bernoulli)
        self.assertEqual(s.node_type(), Tensor)
        self.assertEqual(bmg.add_addition(r, r).node_type(), float)
        self.assertEqual(bmg.add_addition(r, t).node_type(), Tensor)
        self.assertEqual(bmg.add_addition(t, r).node_type(), Tensor)
        self.assertEqual(bmg.add_addition(t, t).node_type(), Tensor)
        self.assertEqual(bmg.add_addition(s, r).node_type(), Tensor)
        self.assertEqual(bmg.add_division(r, r).node_type(), float)
        self.assertEqual(bmg.add_division(r, t).node_type(), Tensor)
        self.assertEqual(bmg.add_division(t, r).node_type(), Tensor)
        self.assertEqual(bmg.add_division(t, t).node_type(), Tensor)
        self.assertEqual(bmg.add_division(s, r).node_type(), Tensor)
        self.assertEqual(bmg.add_multiplication(r, r).node_type(), float)
        self.assertEqual(bmg.add_multiplication(r, t).node_type(), Tensor)
        self.assertEqual(bmg.add_multiplication(t, r).node_type(), Tensor)
        self.assertEqual(bmg.add_multiplication(t, t).node_type(), Tensor)
        self.assertEqual(bmg.add_multiplication(s, r).node_type(), Tensor)
        self.assertEqual(bmg.add_power(r, r).node_type(), float)
        self.assertEqual(bmg.add_power(r, t).node_type(), Tensor)
        self.assertEqual(bmg.add_power(t, r).node_type(), Tensor)
        self.assertEqual(bmg.add_power(t, t).node_type(), Tensor)
        self.assertEqual(bmg.add_power(s, r).node_type(), Tensor)
        self.assertEqual(bmg.add_negate(r).node_type(), float)
        self.assertEqual(bmg.add_negate(t).node_type(), Tensor)
        self.assertEqual(bmg.add_negate(s).node_type(), Tensor)
        self.assertEqual(bmg.add_not(b).node_type(), bool)
        self.assertEqual(bmg.add_not(t).node_type(), bool)
        self.assertEqual(bmg.add_not(s).node_type(), bool)
        self.assertEqual(bmg.add_exp(r).node_type(), float)
        self.assertEqual(bmg.add_exp(t).node_type(), Tensor)
        self.assertEqual(bmg.add_exp(s).node_type(), Tensor)
        self.assertEqual(bmg.add_log(r).node_type(), float)
        self.assertEqual(bmg.add_log(t).node_type(), Tensor)
        self.assertEqual(bmg.add_log(s).node_type(), Tensor)
        self.assertEqual(bmg.add_to_real(t).node_type(), float)
        self.assertEqual(bmg.add_to_real(b).node_type(), float)
        self.assertEqual(bmg.add_to_real(s).node_type(), float)

    def test_sizes(self) -> None:
        bmg = BMGraphBuilder()
        t = bmg.add_tensor(tensor([1.0, 2.0]))
        z1 = bmg.add_tensor(torch.zeros(1, 2))
        z2 = bmg.add_tensor(torch.zeros(2, 1))
        r = bmg.add_real(1.0)
        bern = bmg.add_bernoulli(t)
        s = bmg.add_sample(bern)
        self.assertEqual(t.size, Size([2]))
        self.assertEqual(r.size, Size([]))
        self.assertEqual(bern.size, Size([2]))
        self.assertEqual(s.size, Size([2]))
        self.assertEqual(bmg.add_matrix_multiplication(z1, z2).size, Size([1, 1]))
        self.assertEqual(bmg.add_addition(r, r).size, Size([]))
        self.assertEqual(bmg.add_addition(r, t).size, Size([2]))
        self.assertEqual(bmg.add_addition(t, r).size, Size([2]))
        self.assertEqual(bmg.add_addition(t, t).size, Size([2]))
        self.assertEqual(bmg.add_addition(s, r).size, Size([2]))
        self.assertEqual(bmg.add_division(r, r).size, Size([]))
        self.assertEqual(bmg.add_division(r, t).size, Size([2]))
        self.assertEqual(bmg.add_division(t, r).size, Size([2]))
        self.assertEqual(bmg.add_division(t, t).size, Size([2]))
        self.assertEqual(bmg.add_division(s, r).size, Size([2]))
        self.assertEqual(bmg.add_multiplication(r, r).size, Size([]))
        self.assertEqual(bmg.add_multiplication(r, t).size, Size([2]))
        self.assertEqual(bmg.add_multiplication(t, r).size, Size([2]))
        self.assertEqual(bmg.add_multiplication(t, t).size, Size([2]))
        self.assertEqual(bmg.add_multiplication(s, r).size, Size([2]))
        self.assertEqual(bmg.add_power(r, r).size, Size([]))
        self.assertEqual(bmg.add_power(r, t).size, Size([2]))
        self.assertEqual(bmg.add_power(t, r).size, Size([2]))
        self.assertEqual(bmg.add_power(t, t).size, Size([2]))
        self.assertEqual(bmg.add_power(s, r).size, Size([2]))
        self.assertEqual(bmg.add_negate(r).size, Size([]))
        self.assertEqual(bmg.add_negate(t).size, Size([2]))
        self.assertEqual(bmg.add_negate(s).size, Size([2]))
        self.assertEqual(bmg.add_exp(r).size, Size([]))
        self.assertEqual(bmg.add_exp(t).size, Size([2]))
        self.assertEqual(bmg.add_exp(s).size, Size([2]))
        self.assertEqual(bmg.add_log(r).size, Size([]))
        self.assertEqual(bmg.add_log(t).size, Size([2]))
        self.assertEqual(bmg.add_log(s).size, Size([2]))

    def test_supports(self) -> None:
        bmg = BMGraphBuilder()
        t5 = tensor(0.5)
        t1 = tensor(1.0)
        t2 = tensor(2.0)
        t0 = tensor(0.0)
        t = bmg.add_tensor(t5)
        bern = bmg.add_bernoulli(t)
        s = bmg.add_sample(bern)
        a1 = bmg.add_addition(s, t)
        a2 = bmg.add_addition(s, s)
        self.assertEqual(SetOfTensors(t.support()), SetOfTensors([t5]))
        self.assertEqual(SetOfTensors(s.support()), SetOfTensors([t0, t1]))
        self.assertEqual(SetOfTensors(a1.support()), SetOfTensors([t0 + t5, t1 + t5]))
        self.assertEqual(SetOfTensors(a2.support()), SetOfTensors([t0, t1, t2]))

    def test_maps(self) -> None:
        bmg = BMGraphBuilder()

        t0 = bmg.add_tensor(tensor(0.0))
        t1 = bmg.add_tensor(tensor(1.0))
        t2 = bmg.add_tensor(tensor(2.0))
        t5 = bmg.add_tensor(tensor(0.5))
        bern = bmg.add_bernoulli(t5)
        s1 = bmg.add_sample(bern)
        s2 = bmg.add_sample(bern)
        s3 = bmg.add_sample(bern)
        a = bmg.add_addition(s2, t2)
        m = bmg.add_map(t0, s1, t1, a)
        i = bmg.add_index(m, s3)
        self.assertEqual(
            SetOfTensors(i.support()),
            SetOfTensors([tensor(0.0), tensor(1.0), tensor(2.0), tensor(3.0)]),
        )
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=0.0];
  N10[label=index];
  N1[label=1.0];
  N2[label=2.0];
  N3[label=0.5];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label=Sample];
  N7[label=Sample];
  N8[label="+"];
  N9[label=map];
  N10 -> N7[label=right];
  N10 -> N9[label=left];
  N4 -> N3[label=probability];
  N5 -> N4[label=operand];
  N6 -> N4[label=operand];
  N7 -> N4[label=operand];
  N8 -> N2[label=right];
  N8 -> N6[label=left];
  N9 -> N0[label=0];
  N9 -> N1[label=2];
  N9 -> N5[label=1];
  N9 -> N8[label=3];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_normal(self) -> None:
        bmg = BMGraphBuilder()
        t0 = bmg.add_tensor(tensor(0.0))
        t1 = bmg.add_tensor(tensor(1.0))
        n = bmg.add_normal(t0, t1)
        bmg.add_sample(n)
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N2 -> N0[label=mu];
  N2 -> N1[label=sigma];
  N3 -> N2[label=operand];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_dirichlet(self) -> None:
        bmg = BMGraphBuilder()
        t0 = bmg.add_tensor(tensor([1.0, 2.0, 3.0]))
        d = bmg.add_dirichlet(t0)
        bmg.add_sample(d)
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label="[1.0,2.0,3.0]"];
  N1[label=Dirichlet];
  N2[label=Sample];
  N1 -> N0[label=concentration];
  N2 -> N1[label=operand];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_studentt(self) -> None:
        bmg = BMGraphBuilder()
        df = bmg.add_constant(3.0)
        loc = bmg.add_constant(2.0)
        scale = bmg.add_constant(1.0)
        d = bmg.add_studentt(df, loc, scale)
        bmg.add_sample(d)
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=3.0];
  N1[label=2.0];
  N2[label=1.0];
  N3[label=StudentT];
  N4[label=Sample];
  N3 -> N0[label=df];
  N3 -> N1[label=loc];
  N3 -> N2[label=scale];
  N4 -> N3[label=operand];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_halfcauchy(self) -> None:
        bmg = BMGraphBuilder()
        scale = bmg.add_constant(1.0)
        d = bmg.add_halfcauchy(scale)
        bmg.add_sample(d)
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=1.0];
  N1[label=HalfCauchy];
  N2[label=Sample];
  N1 -> N0[label=scale];
  N2 -> N1[label=operand];
}
"""
        self.assertEqual(observed.strip(), expected.strip())
