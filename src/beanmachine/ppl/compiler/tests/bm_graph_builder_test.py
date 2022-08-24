# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for bm_graph_builder.py"""
import unittest
from typing import Any

import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Tensor


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
        bmg.add_query(mult, RVIdentifier(wrapper=lambda a, b: a, arguments=(1, 1)))

        observed = to_dot(bmg, label_edges=False)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label="Observation True"];
  N4[label=2];
  N5[label=ToReal];
  N6[label="-"];
  N7[label="+"];
  N8[label="*"];
  N9[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N2 -> N5;
  N4 -> N7;
  N4 -> N8;
  N5 -> N6;
  N6 -> N7;
  N7 -> N8;
  N8 -> N9;
}"""
        self.maxDiff = None
        self.assertEqual(expected.strip(), observed.strip())

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
        self.assertEqual(tidy(expected), tidy(observed))

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
n3 = g.add_constant_real(2.0)
n4 = g.add_operator(graph.OperatorType.TO_REAL, [n2])
n5 = g.add_operator(graph.OperatorType.NEGATE, [n4])
n6 = g.add_operator(graph.OperatorType.ADD, [n3, n5])
n7 = g.add_operator(graph.OperatorType.MULTIPLY, [n3, n6])
q0 = g.query(n7)
"""
        self.assertEqual(expected.strip(), observed.strip())

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
        bmg.add_query(lg, RVIdentifier(wrapper=lambda a, b: a, arguments=(1, 1)))
        # Note that the orphan nodes "1" and "4" are not stripped out
        # by default. If you want them gone, the "after_transform" flag does
        # a type check and also removes everything that is not an ancestor
        # of a query or observation.
        observed = to_dot(bmg, label_edges=False)
        expected = """
digraph "graph" {
  N00[label=1];
  N01[label=4];
  N02[label=0.25];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label=complement];
  N06[label=ToPosReal];
  N07[label=2];
  N08[label="/"];
  N09[label="**"];
  N10[label=Log];
  N11[label=Query];
  N02 -> N03;
  N03 -> N04;
  N04 -> N05;
  N05 -> N06;
  N06 -> N08;
  N07 -> N08;
  N07 -> N09;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.maxDiff = None
        self.assertEqual(expected.strip(), observed.strip())

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
        self.assertEqual(tidy(expected), tidy(observed))

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
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=1.0];
  N4[label=0.0];
  N5[label=if];
  N0 -> N1[label=probability];
  N1 -> N2[label=operand];
  N2 -> N5[label=condition];
  N3 -> N5[label=consequence];
  N4 -> N5[label=alternative];
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_allowed_functions(self) -> None:
        bmg = BMGRuntime()
        p = bmg._bmg.add_constant(0.5)
        b = bmg._bmg.add_bernoulli(p)
        s = bmg._bmg.add_sample(b)
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
