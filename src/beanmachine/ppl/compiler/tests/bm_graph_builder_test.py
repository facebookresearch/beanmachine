# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_graph_builder.py"""
import math
import unittest
from typing import Any

import beanmachine.ppl.utils.hint as hint
import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import (
    AdditionNode,
    BernoulliNode,
    BooleanNode,
    ConstantTensorNode,
    DivisionNode,
    EqualNode,
    ExpNode,
    GreaterThanEqualNode,
    GreaterThanNode,
    LessThanEqualNode,
    LessThanNode,
    Log1mexpNode,
    LogNode,
    MatrixMultiplicationNode,
    MultiplicationNode,
    NegateNode,
    NotEqualNode,
    NotNode,
    PowerNode,
    RealNode,
    SampleNode,
    ToRealNode,
)
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
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

    def test_graph_builder_1(self) -> None:

        # Just a trivial model to test whether we can take a properly-typed
        # accumulated graph and turn it into BMG, DOT, or a program that
        # produces a BMG.
        #
        # @random_variable def flip(): return Bernoulli(0.5)
        # @functional      def mult(): return (-flip() + 2) * 2
        bmg = BMGraphBuilder()
        half = bmg.add_probability(0.5)
        two = bmg.add_real(2)
        flip = bmg.add_bernoulli(half)
        samp = bmg.add_sample(flip)
        real = bmg.add_to_real(samp)
        neg = bmg.add_negate(real)
        add = bmg.add_addition(two, neg)
        mult = bmg.add_multiplication(two, add)
        bmg.add_observation(samp, True)
        bmg.add_query(mult)

        observed = to_dot(bmg, label_edges=False)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=2];
  N2[label=Bernoulli];
  N3[label=Sample];
  N4[label=ToReal];
  N5[label="-"];
  N6[label="+"];
  N7[label="*"];
  N8[label="Observation True"];
  N9[label=Query];
  N0 -> N2;
  N1 -> N6;
  N1 -> N7;
  N2 -> N3;
  N3 -> N4;
  N3 -> N8;
  N4 -> N5;
  N5 -> N6;
  N6 -> N7;
  N7 -> N9;
}
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

        g = to_bmg_graph(bmg).graph
        observed = g.to_string()
        expected = """
Node 0 type 1 parents [ ] children [ 1 ] probability 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 4 ] boolean 1
Node 3 type 1 parents [ ] children [ 6 7 ] real 2
Node 4 type 3 parents [ 2 ] children [ 5 ] real 0
Node 5 type 3 parents [ 4 ] children [ 6 ] real 0
Node 6 type 3 parents [ 3 5 ] children [ 7 ] real 0
Node 7 type 3 parents [ 3 6 ] children [ ] real 0"""
        self.assertEqual(tidy(observed), tidy(expected))

        observed = to_bmg_python(bmg).code

        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_probability(0.5)
n1 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n0],
)
n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
g.observe(n2, True)
n3 = g.add_constant(2.0)
n4 = g.add_operator(graph.OperatorType.TO_REAL, [n2])
n5 = g.add_operator(graph.OperatorType.NEGATE, [n4])
n6 = g.add_operator(graph.OperatorType.ADD, [n3, n5])
n7 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n6])
q0 = g.query(n7)
"""
        self.assertEqual(observed.strip(), expected.strip())

        observed = to_bmg_cpp(bmg).code

        expected = """
graph::Graph g;
uint n0 = g.add_constant_probability(0.5);
uint n1 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
g.observe([n2], true);
uint n3 = g.add_constant(2.0);
uint n4 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n2}));
uint n5 = g.add_operator(
  graph::OperatorType::NEGATE, std::vector<uint>({n4}));
uint n6 = g.add_operator(
  graph::OperatorType::ADD, std::vector<uint>({n3, n5}));
uint n7 = g.add_operator(
  graph::OperatorType::MULTIPLY, std::vector<uint>({n3, n6}));
uint q0 = g.query(n7);
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_graph_builder_2(self) -> None:
        bmg = BMGraphBuilder()

        one = bmg.add_pos_real(1)
        two = bmg.add_pos_real(2)
        # These should all be folded:
        four = bmg.add_power(two, two)
        fourth = bmg.add_division(one, four)
        flip = bmg.add_bernoulli(fourth)
        samp = bmg.add_sample(flip)
        inv = bmg.add_complement(samp)  #  NOT operation
        real = bmg.add_to_positive_real(inv)
        div = bmg.add_division(real, two)
        p = bmg.add_power(div, two)
        lg = bmg.add_log(p)
        bmg.add_query(lg)
        # Note that the orphan nodes "1" and "4" are not stripped out
        # by default. If you want them gone, the "after_transform" flag does
        # a type check and also removes everything that is not an ancestor
        # of a query or observation.
        observed = to_dot(bmg, label_edges=False)
        expected = """
digraph "graph" {
  N00[label=1];
  N01[label=2];
  N02[label=4];
  N03[label=0.25];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=complement];
  N07[label=ToPosReal];
  N08[label="/"];
  N09[label="**"];
  N10[label=Log];
  N11[label=Query];
  N01 -> N08;
  N01 -> N09;
  N03 -> N04;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

        g = to_bmg_graph(bmg).graph
        observed = g.to_string()
        # Here however the orphaned nodes are never added to the graph.
        expected = """
Node 0 type 1 parents [ ] children [ 1 ] probability 0.25
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 ] boolean 0
Node 3 type 3 parents [ 2 ] children [ 4 ] boolean 0
Node 4 type 3 parents [ 3 ] children [ 6 ] positive real 1e-10
Node 5 type 1 parents [ ] children [ 6 ] positive real 0.5
Node 6 type 3 parents [ 4 5 ] children [ 8 ] positive real 1e-10
Node 7 type 1 parents [ ] children [ 8 ] positive real 2
Node 8 type 3 parents [ 6 7 ] children [ 9 ] positive real 1e-10
Node 9 type 3 parents [ 8 ] children [ ] real 0"""
        self.assertEqual(tidy(observed), tidy(expected))

    # The "add" methods do exactly that: add a node to the graph if it is not
    # already there.
    #
    # The "handle" methods try to keep everything in unwrapped values if possible;
    # they are trying to keep values out of the graph when possible. More specifically:
    # The arithmetic "handle" functions will detect when they are given normal
    # values and not produce graph nodes; they just do the math and produce the
    # normal Python value. The "handle" functions for constructing distributions
    # however are only ever called in non-test scenarios when a graph node
    # *must* be added to the graph, so they always do so.
    #
    # The next few tests verify that the handle functions are working as designed.

    def test_handle_bernoulli(self) -> None:

        # handle_bernoulli always adds a node to the graph.

        bmg = BMGraphBuilder()

        # In normal operation of the graph accumulator the call below never happens;
        # we only ever call handle_bernoulli if the Bernoulli constructor is invoked
        # with a graph node as an operand. However, it will correctly handle the case
        # where it is given a normal value as an argument, which is useful for
        # testing.

        b = bmg.handle_bernoulli(0.5)
        self.assertTrue(isinstance(b, BernoulliNode))

        # In normal operation of the graph accumulator the call sequence below never
        # happens either, but once again we support this sequence of calls as it
        # is useful for test cases.

        r = bmg.add_real(0.5)
        b = bmg.handle_bernoulli(r)
        self.assertTrue(isinstance(b, BernoulliNode))

        # The sequence that does normally happen is: handle_function has the
        # Bernoulli constructor as the function and a graph node containing a
        # sample (or a node whose ancestor is a sample) and it invokes
        # handle_bernoulli. We can simulate that here:

        beta = bmg.add_beta(r, r)
        betas = bmg.add_sample(beta)
        b = bmg.handle_bernoulli(betas)
        self.assertTrue(isinstance(b, BernoulliNode))

    def test_handle_sample(self) -> None:

        bmg = BMGraphBuilder()

        # Sample on a graph node.
        b = bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5)))
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
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

        # torch defines a "static" add method that takes two values.
        # Calling torch.add(x, y) should be logically the same as x + y

        ta = torch.add
        self.assertEqual(bmg.handle_dot_get(torch, "add"), ta)

        # torch defines an "instance" add method that takes a value.
        # Calling Tensor.add(x, y) or x.add(y) should be logically the same as x + y.

        # In Tensor.add(x, y), x is required to be a tensor, not a double. We do
        # not enforce this rule; handle_function(ta2, [x, y]) where x is a
        # double-valued graph node would not fail.
        #
        # That's fine; ensuring that every Bean Machine program that would crash
        # also crashes during compilation is not a design requirement, particularly
        # when we are generating a BMG model with the desired semantics.

        ta1 = t1.add
        self.assertEqual(bmg.handle_dot_get(t1, "add"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "add")

        ta2 = torch.Tensor.add
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "add"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
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

        # Adding a graph constant and a value produces a value.
        # Note that this does not typically happen during accumulation of
        # a model, but we support it anyways.
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
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))
        gt2 = bmg.add_constant_tensor(t2)
        self.assertTrue(isinstance(gt2, ConstantTensorNode))

        # torch defines a "static" div method that takes two values.
        # Calling torch.div(x, y) should be logically the same as x + y

        ta = torch.div
        self.assertEqual(bmg.handle_dot_get(torch, "div"), ta)

        # torch defines an "instance" div method that takes a value.
        # Calling Tensor.div(x, y) or x.div(y) should be logically the same as x + y.

        ta1 = t2.div
        self.assertEqual(bmg.handle_dot_get(t2, "div"), ta1)

        gta1 = bmg.handle_dot_get(gt2, "div")

        ta2 = torch.Tensor.div
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "div"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
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
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

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
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "exp")

        # Exp of a value produces a value
        self.assertEqual(bmg.handle_exp(1.0), e)
        self.assertEqual(bmg.handle_exp(t1), te)
        self.assertEqual(bmg.handle_function(ta, [t1]), te)
        self.assertEqual(bmg.handle_function(ta1, []), te)
        self.assertEqual(bmg.handle_function(ta2, [t1]), te)

        # Exp of a graph constant produces a value
        self.assertEqual(bmg.handle_exp(gr1), e)
        self.assertEqual(bmg.handle_exp(gt1), te)
        self.assertEqual(bmg.handle_function(ta, [gr1]), e)
        self.assertEqual(bmg.handle_function(ta, [gt1]), te)
        self.assertEqual(bmg.handle_function(gta1, []), te)
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

        # This test verifies that various mechanisms for producing a log node
        # in the graph -- or avoiding producing such a node -- are working as designed.

        bmg = BMGraphBuilder()

        t0 = tensor(0.0)
        t1 = tensor(1.0)

        # Graph nodes
        gr1 = bmg.add_real(1.0)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

        # torch defines a "static" log method that takes one value.
        ta = torch.log
        self.assertEqual(bmg.handle_dot_get(torch, "log"), ta)

        # torch defines an "instance" log method that takes no arguments.
        # Calling Tensor.log(x) or x.log() should produce a log node.

        ta1 = t1.log
        self.assertEqual(bmg.handle_dot_get(t1, "log"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "log")

        ta2 = torch.Tensor.log
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "log"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "log")

        # Log of a value produces a value
        self.assertEqual(bmg.handle_log(1.0), 0.0)
        self.assertEqual(bmg.handle_log(t1), t0)
        self.assertEqual(bmg.handle_function(ta, [t1]), t0)
        self.assertEqual(bmg.handle_function(ta1, []), t0)
        self.assertEqual(bmg.handle_function(ta2, [t1]), t0)

        # Log of a graph constant produces a value
        self.assertEqual(bmg.handle_log(gr1), 0.0)
        self.assertEqual(bmg.handle_log(gt1), t0)
        self.assertEqual(bmg.handle_function(ta, [gr1]), 0.0)
        self.assertEqual(bmg.handle_function(ta, [gt1]), t0)
        self.assertEqual(bmg.handle_function(gta1, []), t0)
        self.assertEqual(bmg.handle_function(ta2, [gr1]), 0.0)
        self.assertEqual(bmg.handle_function(ta2, [gt1]), t0)

        # Log of sample produces node
        n = LogNode
        self.assertTrue(isinstance(bmg.handle_log(s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_log1mexp(self) -> None:
        """Test log1mexp - based on test_log."""

        # Kicking the tires on log1mexp

        bmg = BMGraphBuilder()

        # Let's pick a pair of values such as v0 = log1mexp(v1)
        v0 = -0.45867514538708193
        v1 = -1.0

        # Now we get the corresponding values as vectors
        t0 = tensor(v0)
        t1 = tensor(v1)

        # Graph nodes corresponding to t1
        gr1 = bmg.add_real(v1)
        self.assertTrue(isinstance(gr1, RealNode))
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

        # utils.hint defines a "static" log1mexp method that takes one value.
        ta = hint.log1mexp
        self.assertEqual(bmg.handle_dot_get(hint, "log1mexp"), ta)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        # Log of a value produces a value
        self.assertEqual(bmg.handle_log1mexp(v1), v0)
        self.assertEqual(bmg.handle_log1mexp(t1), t0)
        self.assertEqual(bmg.handle_function(ta, [t1]), t0)

        # Log of a graph constant produces a value
        self.assertEqual(bmg.handle_log1mexp(gr1), v0)
        self.assertEqual(bmg.handle_log1mexp(gt1), t0)
        self.assertEqual(bmg.handle_function(ta, [gr1]), v0)
        self.assertEqual(bmg.handle_function(ta, [gt1]), t0)

        # Log of sample produces node
        n = Log1mexpNode
        self.assertTrue(isinstance(bmg.handle_log1mexp(s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s]), n))

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
        gr2 = bmg.add_real(2.0)
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))
        gt2 = bmg.add_constant_tensor(t2)

        # torch defines a "static" mul method that takes two values.
        # Calling torch.mul(x, y) should be logically the same as x * y

        ta = torch.mul
        self.assertEqual(bmg.handle_dot_get(torch, "mul"), ta)

        # torch defines an "instance" mul method that takes a value.
        # Calling Tensor.mul(x, y) or x.mul(y) should be logically the same as x * y.
        ta1 = t1.mul
        self.assertEqual(bmg.handle_dot_get(t1, "mul"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "mul")
        gta2 = bmg.handle_dot_get(gt2, "mul")

        ta2 = torch.Tensor.mul
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "mul"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
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
        self.assertTrue(isinstance(bmg.handle_function(gta2, [s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, 2.0]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, t2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [t2, s]), n))

        # Sample times graph node produces node
        self.assertTrue(isinstance(bmg.handle_multiplication(s, gr2), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(s, gt2), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(gr2, s), n))
        self.assertTrue(isinstance(bmg.handle_multiplication(gt2, s), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gr2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [s, gt2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gr2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta, [gt2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gr2]), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, [gt2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gr2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s, gt2]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt2, s]), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [gt2], {"other": s}), n))

    def test_matrix_multiplication(self) -> None:
        """Test matrix_multiplication"""

        # This test verifies that various mechanisms for producing a matrix
        # multiplication node in the graph -- or avoiding producing such a
        # node -- are working as designed.

        bmg = BMGraphBuilder()

        t1 = tensor([[3.0, 4.0], [5.0, 6.0]])
        t2 = tensor([[29.0, 36.0], [45.0, 56.0]])

        # Graph nodes
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))
        gt2 = bmg.add_constant_tensor(t2)
        self.assertTrue(isinstance(gt2, ConstantTensorNode))

        # torch defines a "static" mm method that takes two values.

        ta = torch.mm
        self.assertEqual(bmg.handle_dot_get(torch, "mm"), ta)

        # torch defines an "instance" mm method that takes a value.

        ta1 = t1.mm
        self.assertEqual(bmg.handle_dot_get(t1, "mm"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "mm")

        ta2 = torch.Tensor.mm
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "mm"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(
            bmg.add_bernoulli(bmg.add_constant_tensor(tensor([[0.5, 0.5], [0.5, 0.5]])))
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

    def test_comparison(self) -> None:
        """Test comparison"""
        bmg = BMGraphBuilder()

        t1 = tensor(1.0)
        t2 = tensor(2.0)
        f = tensor(False)
        t = tensor(True)
        g1 = bmg.add_real(1.0)
        g2 = bmg.add_real(2.0)
        s = bmg.add_sample(bmg.add_halfcauchy(bmg.add_real(0.5)))

        self.assertEqual(bmg.handle_equal(t2, t1), f)
        self.assertEqual(bmg.handle_equal(t1, t1), t)
        self.assertEqual(bmg.handle_equal(t1, t2), f)
        self.assertEqual(bmg.handle_equal(g2, t1), f)
        self.assertEqual(bmg.handle_equal(g1, t1), t)
        self.assertEqual(bmg.handle_equal(g1, t2), f)
        self.assertEqual(bmg.handle_equal(t2, g1), f)
        self.assertEqual(bmg.handle_equal(t1, g1), t)
        self.assertEqual(bmg.handle_equal(t1, g2), f)
        self.assertEqual(bmg.handle_equal(g2, g1), f)
        self.assertEqual(bmg.handle_equal(g1, g1), t)
        self.assertEqual(bmg.handle_equal(g1, g2), f)
        self.assertTrue(isinstance(bmg.handle_equal(s, t1), EqualNode))
        self.assertTrue(isinstance(bmg.handle_equal(t1, s), EqualNode))

        self.assertEqual(bmg.handle_not_equal(t2, t1), t)
        self.assertEqual(bmg.handle_not_equal(t1, t1), f)
        self.assertEqual(bmg.handle_not_equal(t1, t2), t)
        self.assertEqual(bmg.handle_not_equal(g2, t1), t)
        self.assertEqual(bmg.handle_not_equal(g1, t1), f)
        self.assertEqual(bmg.handle_not_equal(g1, t2), t)
        self.assertEqual(bmg.handle_not_equal(t2, g1), t)
        self.assertEqual(bmg.handle_not_equal(t1, g1), f)
        self.assertEqual(bmg.handle_not_equal(t1, g2), t)
        self.assertEqual(bmg.handle_not_equal(g2, g1), t)
        self.assertEqual(bmg.handle_not_equal(g1, g1), f)
        self.assertEqual(bmg.handle_not_equal(g1, g2), t)
        self.assertTrue(isinstance(bmg.handle_not_equal(s, t1), NotEqualNode))
        self.assertTrue(isinstance(bmg.handle_not_equal(t1, s), NotEqualNode))

        self.assertEqual(bmg.handle_greater_than(t2, t1), t)
        self.assertEqual(bmg.handle_greater_than(t1, t1), f)
        self.assertEqual(bmg.handle_greater_than(t1, t2), f)
        self.assertEqual(bmg.handle_greater_than(g2, t1), t)
        self.assertEqual(bmg.handle_greater_than(g1, t1), f)
        self.assertEqual(bmg.handle_greater_than(g1, t2), f)
        self.assertEqual(bmg.handle_greater_than(t2, g1), t)
        self.assertEqual(bmg.handle_greater_than(t1, g1), f)
        self.assertEqual(bmg.handle_greater_than(t1, g2), f)
        self.assertEqual(bmg.handle_greater_than(g2, g1), t)
        self.assertEqual(bmg.handle_greater_than(g1, g1), f)
        self.assertEqual(bmg.handle_greater_than(g1, g2), f)
        self.assertTrue(isinstance(bmg.handle_greater_than(s, t1), GreaterThanNode))
        self.assertTrue(isinstance(bmg.handle_greater_than(t1, s), GreaterThanNode))

        self.assertEqual(bmg.handle_greater_than_equal(t2, t1), t)
        self.assertEqual(bmg.handle_greater_than_equal(t1, t1), t)
        self.assertEqual(bmg.handle_greater_than_equal(t1, t2), f)
        self.assertEqual(bmg.handle_greater_than_equal(g2, t1), t)
        self.assertEqual(bmg.handle_greater_than_equal(g1, t1), t)
        self.assertEqual(bmg.handle_greater_than_equal(g1, t2), f)
        self.assertEqual(bmg.handle_greater_than_equal(t2, g1), t)
        self.assertEqual(bmg.handle_greater_than_equal(t1, g1), t)
        self.assertEqual(bmg.handle_greater_than_equal(t1, g2), f)
        self.assertEqual(bmg.handle_greater_than_equal(g2, g1), t)
        self.assertEqual(bmg.handle_greater_than_equal(g1, g1), t)
        self.assertEqual(bmg.handle_greater_than_equal(g1, g2), f)
        self.assertTrue(
            isinstance(bmg.handle_greater_than_equal(s, t1), GreaterThanEqualNode)
        )
        self.assertTrue(
            isinstance(bmg.handle_greater_than_equal(t1, s), GreaterThanEqualNode)
        )

        self.assertEqual(bmg.handle_less_than(t2, t1), f)
        self.assertEqual(bmg.handle_less_than(t1, t1), f)
        self.assertEqual(bmg.handle_less_than(t1, t2), t)
        self.assertEqual(bmg.handle_less_than(g2, t1), f)
        self.assertEqual(bmg.handle_less_than(g1, t1), f)
        self.assertEqual(bmg.handle_less_than(g1, t2), t)
        self.assertEqual(bmg.handle_less_than(t2, g1), f)
        self.assertEqual(bmg.handle_less_than(t1, g1), f)
        self.assertEqual(bmg.handle_less_than(t1, g2), t)
        self.assertEqual(bmg.handle_less_than(g2, g1), f)
        self.assertEqual(bmg.handle_less_than(g1, g1), f)
        self.assertEqual(bmg.handle_less_than(g1, g2), t)
        self.assertTrue(isinstance(bmg.handle_less_than(s, t1), LessThanNode))
        self.assertTrue(isinstance(bmg.handle_less_than(t1, s), LessThanNode))

        self.assertEqual(bmg.handle_less_than_equal(t2, t1), f)
        self.assertEqual(bmg.handle_less_than_equal(t1, t1), t)
        self.assertEqual(bmg.handle_less_than_equal(t1, t2), t)
        self.assertEqual(bmg.handle_less_than_equal(g2, t1), f)
        self.assertEqual(bmg.handle_less_than_equal(g1, t1), t)
        self.assertEqual(bmg.handle_less_than_equal(g1, t2), t)
        self.assertEqual(bmg.handle_less_than_equal(t2, g1), f)
        self.assertEqual(bmg.handle_less_than_equal(t1, g1), t)
        self.assertEqual(bmg.handle_less_than_equal(t1, g2), t)
        self.assertEqual(bmg.handle_less_than_equal(g2, g1), f)
        self.assertEqual(bmg.handle_less_than_equal(g1, g1), t)
        self.assertEqual(bmg.handle_less_than_equal(g1, g2), t)
        self.assertTrue(
            isinstance(bmg.handle_less_than_equal(s, t1), LessThanEqualNode)
        )
        self.assertTrue(
            isinstance(bmg.handle_less_than_equal(t1, s), LessThanEqualNode)
        )

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
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

        # torch defines a "static" neg method that takes one value.
        # Calling torch.neg(x) should be logically the same as -x
        ta = torch.neg
        self.assertEqual(bmg.handle_dot_get(torch, "neg"), ta)

        # torch defines an "instance" neg method that takes no arguments.
        # Calling Tensor.neg(x) or x.neg() should be logically the same as -x.
        ta1 = t1.neg
        self.assertEqual(bmg.handle_dot_get(t1, "neg"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "neg")

        ta2 = torch.Tensor.neg
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "neg"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
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
        gtt = bmg.add_constant_tensor(tt)
        self.assertTrue(isinstance(gtt, ConstantTensorNode))

        # torch defines a "static" logical_not method that takes one value.
        # Calling torch.logical_not(x) should be logically the same as "not x"
        ta = torch.logical_not
        self.assertEqual(bmg.handle_dot_get(torch, "logical_not"), ta)

        # torch defines an "instance" add method that takes no arguments.
        # Calling Tensor.logical_not(x) or x.logical_not() should be logically
        # the same as "not x".
        ta1 = tt.logical_not
        self.assertEqual(bmg.handle_dot_get(tt, "logical_not"), ta1)

        gta1 = bmg.handle_dot_get(gtt, "logical_not")

        ta2 = torch.Tensor.logical_not
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "logical_not"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
        self.assertTrue(isinstance(s, SampleNode))

        sa = bmg.handle_dot_get(s, "logical_not")

        # Negating a value produces a value
        self.assertEqual(bmg.handle_not(True), False)
        self.assertEqual(bmg.handle_not(tt), tf)
        self.assertEqual(bmg.handle_function(ta, [tt]), tf)
        self.assertEqual(bmg.handle_function(ta1, []), tf)
        self.assertEqual(bmg.handle_function(ta2, [tt]), tf)

        # Negating a graph constant produces a value
        self.assertEqual(bmg.handle_not(gbt), False)
        self.assertEqual(bmg.handle_not(gtt), tf)
        self.assertEqual(bmg.handle_function(ta, [gbt]), False)
        self.assertEqual(bmg.handle_function(ta, [gtt]), tf)
        self.assertEqual(bmg.handle_function(ta, [], {"input": gtt}), tf)
        self.assertEqual(bmg.handle_function(gta1, []), tf)
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
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

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

        ta1 = t1.pow
        self.assertEqual(bmg.handle_dot_get(t1, "pow"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "pow")

        ta2 = torch.Tensor.pow
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "pow"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
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
        gt1 = bmg.add_constant_tensor(t1)
        self.assertTrue(isinstance(gt1, ConstantTensorNode))

        # torch.float is not a function, unlike torch.log, torch.add and so on.

        # torch defines an "instance" float method that takes no arguments.
        # Calling Tensor.float(x) or x.float() should produce a to_real node.

        ta1 = t1.float
        self.assertEqual(bmg.handle_dot_get(t1, "float"), ta1)

        gta1 = bmg.handle_dot_get(gt1, "float")

        ta2 = torch.Tensor.float
        self.assertEqual(bmg.handle_dot_get(torch.Tensor, "float"), ta2)

        # Make a sample node; this cannot be simplified away.
        s = bmg.add_sample(bmg.add_bernoulli(bmg.add_constant_tensor(tensor(0.5))))
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
        self.assertEqual(bmg.handle_function(ta2, [gr1]), 1.0)
        self.assertEqual(bmg.handle_function(ta2, [gt1]), 1.0)

        # Float of sample produces node
        n = ToRealNode
        self.assertTrue(isinstance(bmg.handle_to_real(s), n))
        self.assertTrue(isinstance(bmg.handle_function(sa, []), n))
        self.assertTrue(isinstance(bmg.handle_function(ta2, [s]), n))

    def test_to_positive_real(self) -> None:
        """Test to_positive_real"""
        bmg = BMGraphBuilder()
        two = bmg.add_pos_real(2.0)
        # to_positive_real on a positive real constant is an identity
        self.assertEqual(bmg.add_to_positive_real(two), two)
        beta22 = bmg.add_beta(two, two)
        to_pr = bmg.add_to_positive_real(beta22)
        # to_positive_real nodes are deduplicated
        self.assertEqual(bmg.add_to_positive_real(beta22), to_pr)

    def test_to_probability(self) -> None:
        """Test to_probability"""
        bmg = BMGraphBuilder()
        h = bmg.add_probability(0.5)
        # to_probability on a prob constant is an identity
        self.assertEqual(bmg.add_to_probability(h), h)
        # We have (hc / (0.5 + hc)) which is always between
        # 0 and 1, but the quotient of two positive reals
        # is a positive real. Force it to be a probability.
        hc = bmg.add_halfcauchy(h)
        s = bmg.add_addition(hc, h)
        q = bmg.add_division(hc, s)
        to_p = bmg.add_to_probability(q)
        # to_probability nodes are deduplicated
        self.assertEqual(bmg.add_to_probability(q), to_p)

    def test_sizes(self) -> None:
        bmg = BMGraphBuilder()
        t = bmg.add_constant_tensor(tensor([1.0, 2.0]))
        z1 = bmg.add_constant_tensor(torch.zeros(1, 2))
        z2 = bmg.add_constant_tensor(torch.zeros(2, 1))
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
        nr = bmg.add_real(-1.0)
        nt = bmg.add_constant_tensor(tensor([-1.0, -2.0]))
        self.assertEqual(bmg.add_log1mexp(nr).size, Size([]))
        self.assertEqual(bmg.add_log1mexp(nt).size, Size([2]))
        self.assertEqual(bmg.add_log1mexp(s).size, Size([2]))

    def test_if_then_else(self) -> None:
        self.maxDiff = None
        bmg = BMGraphBuilder()
        p = bmg.add_constant(0.5)
        z = bmg.add_constant(0.0)
        o = bmg.add_constant(1.0)
        b = bmg.add_bernoulli(p)
        s = bmg.add_sample(b)
        bmg.add_if_then_else(s, o, z)
        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=0.0];
  N2[label=1.0];
  N3[label=Bernoulli];
  N4[label=Sample];
  N5[label=if];
  N0 -> N3[label=probability];
  N1 -> N5[label=alternative];
  N2 -> N5[label=consequence];
  N3 -> N4[label=operand];
  N4 -> N5[label=condition];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_allowed_functions(self) -> None:
        bmg = BMGraphBuilder()
        p = bmg.add_constant(0.5)
        b = bmg.add_bernoulli(p)
        s = bmg.add_sample(b)
        d = bmg.handle_function(dict, [[(1, s)]])
        self.assertEqual(d, {1: s})

    def test_add_tensor(self) -> None:
        bmg = BMGraphBuilder()
        p = bmg.add_constant(0.5)
        b = bmg.add_bernoulli(p)
        s = bmg.add_sample(b)

        # Tensors are deduplicated
        t1 = bmg.add_tensor(torch.Size([3]), s, s, p)
        t2 = bmg.add_tensor(torch.Size([3]), *[s, s, p])
        self.assertTrue(t1 is t2)

        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Tensor];
  N0 -> N1[label=probability];
  N0 -> N3[label=2];
  N1 -> N2[label=operand];
  N2 -> N3[label=0];
  N2 -> N3[label=1];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_remove_leaf_from_builder(self) -> None:
        bmg = BMGraphBuilder()
        p = bmg.add_constant(0.5)
        b = bmg.add_bernoulli(p)
        s = bmg.add_sample(b)
        o = bmg.add_observation(s, True)

        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label="Observation True"];
  N0 -> N1[label=probability];
  N1 -> N2[label=operand];
  N2 -> N3[label=operand];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

        with self.assertRaises(ValueError):
            # Not a leaf
            bmg.remove_leaf(s)

        bmg.remove_leaf(o)
        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N0 -> N1[label=probability];
  N1 -> N2[label=operand];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

        # Is a leaf now.
        bmg.remove_leaf(s)
        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N0 -> N1[label=probability];
}
"""
        self.assertEqual(observed.strip(), expected.strip())
