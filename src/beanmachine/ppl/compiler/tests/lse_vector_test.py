# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot


class LSEVectorTest(unittest.TestCase):
    def test_lse_vector_1(self) -> None:

        self.maxDiff = None
        bmg = BMGraphBuilder()
        t = bmg.add_natural(2)
        o = bmg.add_natural(1)
        z = bmg.add_natural(0)
        n = bmg.add_normal(z, o)
        ns = bmg.add_sample(n)
        e = bmg.add_exp(ns)
        # LogSumExpVector requires one column, n rows.
        m = bmg.add_to_matrix(t, o, e, ns)
        lsev = bmg.add_logsumexp_vector(m)
        bmg.add_query(lsev)

        observed = to_dot(
            bmg,
            inf_types=True,
            edge_requirements=True,
            after_transform=True,
            label_edges=True,
        )
        expected = """
digraph "graph" {
  N00[label="0.0:R"];
  N01[label="1.0:R+"];
  N02[label="Normal:R"];
  N03[label="Sample:R"];
  N04[label="2:N"];
  N05[label="1:N"];
  N06[label="Exp:R+"];
  N07[label="ToReal:R"];
  N08[label="ToMatrix:MR[2,1]"];
  N09[label="LogSumExp:R"];
  N10[label="Query:R"];
  N00 -> N02[label="mu:R"];
  N01 -> N02[label="sigma:R+"];
  N02 -> N03[label="operand:R"];
  N03 -> N06[label="operand:R"];
  N03 -> N08[label="1:R"];
  N04 -> N08[label="rows:N"];
  N05 -> N08[label="columns:N"];
  N06 -> N07[label="operand:<=R"];
  N07 -> N08[label="0:R"];
  N08 -> N09[label="operand:MR[2,1]"];
  N09 -> N10[label="operator:any"];
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
uint n4 = g.add_constant(2);
uint n5 = g.add_constant(1);
uint n6 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n3}));
uint n7 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n6}));
uint n8 = g.add_operator(
  graph::OperatorType::TO_MATRIX,
  std::vector<uint>({n4, n5, n7, n3}));
uint n9 = g.add_operator(
  graph::OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>({n8}));
uint q0 = g.query(n9);
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
n4 = g.add_constant(2)
n5 = g.add_constant(1)
n6 = g.add_operator(graph.OperatorType.EXP, [n3])
n7 = g.add_operator(graph.OperatorType.TO_REAL, [n6])
n8 = g.add_operator(
  graph.OperatorType.TO_MATRIX,
  [n4, n5, n7, n3],
)
n9 = g.add_operator(graph.OperatorType.LOGSUMEXP_VECTOR, [n8])
q0 = g.query(n9)
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_graph(bmg).graph.to_dot()
        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="2"];
  N5[label="1"];
  N6[label="exp"];
  N7[label="ToReal"];
  N8[label="ToMatrix"];
  N9[label="LogSumExp"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N6;
  N3 -> N8;
  N4 -> N8;
  N5 -> N8;
  N6 -> N7;
  N7 -> N8;
  N8 -> N9;
  Q0[label="Query"];
  N9 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
