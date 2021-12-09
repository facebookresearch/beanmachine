# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch import tensor
from torch.distributions import Bernoulli, Normal


@bm.random_variable
def norm():
    return Normal(tensor(0.0), tensor(1.0))


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def f1by2():
    # A 1x2 tensor in Python becomes a 2x1 matrix in BMG
    t = tensor([norm().exp(), norm()])
    # This should become a LOGSUMEXP BMG node with no TO_MATRIX
    return t.logsumexp(dim=0)


@bm.functional
def f2by1():
    # A 2x1 tensor in Python becomes a 1x2 matrix in BMG
    t = tensor([[norm().exp()], [norm()]])
    # This should be an error; BMG requires that the matrix have a single column.
    return t.logsumexp(dim=0)


@bm.functional
def f2by3():
    # A 2x3 tensor in Python becomes a 3x2 matrix in BMG
    t = tensor([[norm().exp(), 10, 20], [norm(), 30, 40]])
    # Randomly choose one of the two columns and LSE it.
    # This should become an LOGSUMEXP_VECTOR node.
    return t[flip()].logsumexp(dim=0)


class LSEVectorTest(unittest.TestCase):
    def test_lse1by2(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f1by2()], {})
        observed = to_dot(bmg, after_transform=True, label_edges=False)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Exp];
  N5[label=ToReal];
  N6[label=LogSumExp];
  N7[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N3 -> N6;
  N4 -> N5;
  N5 -> N6;
  N6 -> N7;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_python(bmg).code
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant_real(0.0)
n1 = g.add_constant_pos_real(1.0)
n2 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n0, n1],
)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_operator(graph.OperatorType.EXP, [n3])
n5 = g.add_operator(graph.OperatorType.TO_REAL, [n4])
n6 = g.add_operator(graph.OperatorType.LOGSUMEXP, [n5, n3])
q0 = g.query(n6)
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_graph(bmg).graph.to_dot()
        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="exp"];
  N5[label="ToReal"];
  N6[label="LogSumExp"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N3 -> N6;
  N4 -> N5;
  N5 -> N6;
  Q0[label="Query"];
  N6 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_lse2by3(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f2by3()], {})
        observed = to_dot(bmg, after_transform=True, label_edges=False)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=0.5];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=3];
  N08[label=2];
  N09[label=Exp];
  N10[label=ToReal];
  N11[label=10.0];
  N12[label=20.0];
  N13[label=30.0];
  N14[label=40.0];
  N15[label=ToMatrix];
  N16[label=1];
  N17[label=0];
  N18[label=if];
  N19[label=ColumnIndex];
  N20[label=LogSumExp];
  N21[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N03 -> N09;
  N03 -> N15;
  N04 -> N05;
  N05 -> N06;
  N06 -> N18;
  N07 -> N15;
  N08 -> N15;
  N09 -> N10;
  N10 -> N15;
  N11 -> N15;
  N12 -> N15;
  N13 -> N15;
  N14 -> N15;
  N15 -> N19;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_cpp(bmg).code
        expected = """
graph::Graph g;
uint n0 = g.add_constant(0.0);
uint n1 = g.add_constant_pos_real(1.0);
uint n2 = g.add_distribution(
  graph::DistributionType::NORMAL,
  graph::AtomicType::REAL,
  std::vector<uint>({n0, n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n2}));
uint n4 = g.add_constant_probability(0.5);
uint n5 = g.add_distribution(
  graph::DistributionType::BERNOULLI,
  graph::AtomicType::BOOLEAN,
  std::vector<uint>({n4}));
uint n6 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n5}));
uint n7 = g.add_constant(3);
uint n8 = g.add_constant(2);
uint n9 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n3}));
uint n10 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n9}));
uint n11 = g.add_constant(10.0);
uint n12 = g.add_constant(20.0);
uint n13 = g.add_constant(30.0);
uint n14 = g.add_constant(40.0);
uint n15 = g.add_operator(
  graph::OperatorType::TO_MATRIX,
  std::vector<uint>({n7, n8, n10, n11, n12, n3, n13, n14}));
uint n16 = g.add_constant(1);
uint n17 = g.add_constant(0);
uint n18 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n6, n16, n17}));
uint n19 = g.add_operator(
  graph::OperatorType::COLUMN_INDEX, std::vector<uint>({n15, n18}));
uint n20 = g.add_operator(
  graph::OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>({n19}));
uint q0 = g.query(n20);
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_graph(bmg).graph.to_dot()
        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="0.5"];
  N5[label="Bernoulli"];
  N6[label="~"];
  N7[label="3"];
  N8[label="2"];
  N9[label="exp"];
  N10[label="ToReal"];
  N11[label="10"];
  N12[label="20"];
  N13[label="30"];
  N14[label="40"];
  N15[label="ToMatrix"];
  N16[label="1"];
  N17[label="0"];
  N18[label="IfThenElse"];
  N19[label="ColumnIndex"];
  N20[label="LogSumExp"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N9;
  N3 -> N15;
  N4 -> N5;
  N5 -> N6;
  N6 -> N18;
  N7 -> N15;
  N8 -> N15;
  N9 -> N10;
  N10 -> N15;
  N11 -> N15;
  N12 -> N15;
  N13 -> N15;
  N14 -> N15;
  N15 -> N19;
  N16 -> N18;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  Q0[label="Query"];
  N20 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_lse2by1(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f2by1()], {})
        expected = """
The model uses a LogSumExp operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
"""
        with self.assertRaises(ValueError) as ex:
            to_dot(bmg, after_transform=True)
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())
