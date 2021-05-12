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

    def test_to_matrix_2(self) -> None:

        # Test TO_MATRIX, TO_REAL_MATRIX and TO_POS_REAL_MATRIX.
        # The first composes a matrix from elements; the latter
        # convert a matrix of one type (probability in this case)
        # to a matrix of another type.
        #
        # Notice that we do not explicitly insert a ToRealMatrix
        # node here; the problem fixer detects that we have a 2x1
        # probability matrix from the column index but the
        # LogSumExpVector needs a real, positive real or negative
        # real matrix, and inserts a ToRealMatrix node on that edge.

        self.maxDiff = None
        bmg = BMGraphBuilder()
        zero = bmg.add_constant(0)
        one = bmg.add_constant(1)
        two = bmg.add_natural(2)
        three = bmg.add_constant(3)
        beta = bmg.add_beta(three, three)
        b0 = bmg.add_sample(beta)
        b1 = bmg.add_sample(beta)
        b2 = bmg.add_sample(beta)
        b3 = bmg.add_sample(beta)
        pm = bmg.add_to_matrix(two, two, b0, b1, b2, b3)
        c0 = bmg.add_column_index(pm, zero)
        c1 = bmg.add_column_index(pm, one)
        tpr = bmg.add_to_positive_real_matrix(c1)
        lse0 = bmg.add_logsumexp_vector(c0)
        lse1 = bmg.add_logsumexp_vector(tpr)
        bmg.add_query(lse0)
        bmg.add_query(lse1)

        observed = to_dot(
            bmg,
            inf_types=True,
            edge_requirements=True,
            after_transform=True,
            label_edges=True,
        )
        expected = """
digraph "graph" {
  N00[label="3.0:R+"];
  N01[label="Beta:P"];
  N02[label="Sample:P"];
  N03[label="Sample:P"];
  N04[label="Sample:P"];
  N05[label="Sample:P"];
  N06[label="2:N"];
  N07[label="ToMatrix:MP[2,2]"];
  N08[label="0:N"];
  N09[label="ColumnIndex:MP[2,1]"];
  N10[label="ToRealMatrix:MR[2,1]"];
  N11[label="LogSumExp:R"];
  N12[label="Query:R"];
  N13[label="1:N"];
  N14[label="ColumnIndex:MP[2,1]"];
  N15[label="ToPosRealMatrix:MR+[2,1]"];
  N16[label="LogSumExp:R"];
  N17[label="Query:R"];
  N00 -> N01[label="alpha:R+"];
  N00 -> N01[label="beta:R+"];
  N01 -> N02[label="operand:P"];
  N01 -> N03[label="operand:P"];
  N01 -> N04[label="operand:P"];
  N01 -> N05[label="operand:P"];
  N02 -> N07[label="0:P"];
  N03 -> N07[label="1:P"];
  N04 -> N07[label="2:P"];
  N05 -> N07[label="3:P"];
  N06 -> N07[label="columns:N"];
  N06 -> N07[label="rows:N"];
  N07 -> N09[label="left:MP[2,2]"];
  N07 -> N14[label="left:MP[2,2]"];
  N08 -> N09[label="right:N"];
  N09 -> N10[label="operand:any"];
  N10 -> N11[label="operand:MR[2,1]"];
  N11 -> N12[label="operator:any"];
  N13 -> N14[label="right:N"];
  N14 -> N15[label="operand:any"];
  N15 -> N16[label="operand:MR+[2,1]"];
  N16 -> N17[label="operator:any"];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_cpp(bmg).code
        expected = """
graph::Graph g;
uint n0 = g.add_constant_pos_real(3.0);
uint n1 = g.add_distribution(
  graph::DistributionType::BETA,
  graph::AtomicType::PROBABILITY,
  std::vector<uint>({n0, n0}));
uint n2 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n3 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n4 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n5 = g.add_operator(
  graph::OperatorType::SAMPLE, std::vector<uint>({n1}));
uint n6 = g.add_constant(2);
uint n7 = g.add_operator(
  graph::OperatorType::TO_MATRIX,
  std::vector<uint>({n6, n6, n2, n3, n4, n5}));
uint n8 = g.add_constant(0);
uint n9 = g.add_operator(
  graph::OperatorType::COLUMN_INDEX, std::vector<uint>({n7, n8}));
uint n10 = g.add_operator(
  graph::OperatorType::TO_REAL_MATRIX, std::vector<uint>({n9}));
uint n11 = g.add_operator(
  graph::OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>({n10}));
uint q0 = g.query(n11);
uint n12 = g.add_constant(1);
uint n13 = g.add_operator(
  graph::OperatorType::COLUMN_INDEX, std::vector<uint>({n7, n12}));
uint n14 = g.add_operator(
  graph::OperatorType::TO_POS_REAL_MATRIX, std::vector<uint>({n13}));
uint n15 = g.add_operator(
  graph::OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>({n14}));
uint q1 = g.query(n15);
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = to_bmg_graph(bmg).graph.to_dot()
        expected = """
digraph "graph" {
  N0[label="3"];
  N1[label="Beta"];
  N2[label="~"];
  N3[label="~"];
  N4[label="~"];
  N5[label="~"];
  N6[label="2"];
  N7[label="ToMatrix"];
  N8[label="0"];
  N9[label="ColumnIndex"];
  N10[label="ToReal"];
  N11[label="LogSumExp"];
  N12[label="1"];
  N13[label="ColumnIndex"];
  N14[label="ToPosReal"];
  N15[label="LogSumExp"];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N1 -> N4;
  N1 -> N5;
  N2 -> N7;
  N3 -> N7;
  N4 -> N7;
  N5 -> N7;
  N6 -> N7;
  N6 -> N7;
  N7 -> N9;
  N7 -> N13;
  N8 -> N9;
  N9 -> N10;
  N10 -> N11;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  Q0[label="Query"];
  N11 -> Q0;
  Q1[label="Query"];
  N15 -> Q1;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
