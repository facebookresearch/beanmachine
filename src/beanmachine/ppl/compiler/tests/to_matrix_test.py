# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot


class ToMatrixTest(unittest.TestCase):
    def test_to_matrix_1(self) -> None:

        self.maxDiff = None
        bmg = BMGraphBuilder()
        t = bmg.add_natural(2)
        o = bmg.add_natural(1)
        z = bmg.add_natural(0)
        n = bmg.add_normal(z, o)
        ns = bmg.add_sample(n)
        e = bmg.add_exp(ns)
        m = bmg.add_to_matrix(o, t, e, ns)
        bmg.add_query(m)

        observed = to_dot(
            bmg,
            inf_types=True,
            edge_requirements=True,
            after_transform=True,
            label_edges=True,
        )
        expected = """
digraph "graph" {
  N0[label="0.0:R"];
  N1[label="1.0:R+"];
  N2[label="Normal:R"];
  N3[label="Sample:R"];
  N4[label="1:N"];
  N5[label="2:N"];
  N6[label="Exp:R+"];
  N7[label="ToReal:R"];
  N8[label="ToMatrix:MR[1,2]"];
  N9[label="Query:MR[1,2]"];
  N0 -> N2[label="mu:R"];
  N1 -> N2[label="sigma:R+"];
  N2 -> N3[label="operand:R"];
  N3 -> N6[label="operand:R"];
  N3 -> N8[label="1:R"];
  N4 -> N8[label="rows:N"];
  N5 -> N8[label="columns:N"];
  N6 -> N7[label="operand:<=R"];
  N7 -> N8[label="0:R"];
  N8 -> N9[label="operator:any"];
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
uint n4 = g.add_constant(1);
uint n5 = g.add_constant(2);
uint n6 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n3}));
uint n7 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::TO_MATRIX,
  std::vector<uint>({n4, n5, n7, n3}));
uint q0 = g.query(n8);
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_python(bmg).code
        expected = """
from beanmachine import graph
from torch import tensor
g = graph.Graph()
n0 = g.add_constant(0.0)
n1 = g.add_constant_pos_real(1.0)
n2 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n0, n1],
)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_constant(1)
n5 = g.add_constant(2)
n6 = g.add_operator(graph.OperatorType.EXP, [n3])
n7 = g.add_operator(graph.OperatorType.TO_REAL, [n6])
n8 = g.add_operator(
  graph.OperatorType.TO_MATRIX,
  [n4, n5, n7, n3],
)
q0 = g.query(n8)
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_graph(bmg).graph.to_dot()
        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="1"];
  N5[label="2"];
  N6[label="exp"];
  N7[label="ToReal"];
  N8[label="ToMatrix"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N6;
  N3 -> N8;
  N4 -> N8;
  N5 -> N8;
  N6 -> N7;
  N7 -> N8;
  Q0[label="Query"];
  N8 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
