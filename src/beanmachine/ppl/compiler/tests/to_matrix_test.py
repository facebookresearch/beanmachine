# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import tensor
from torch.distributions import Normal


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


@bm.random_variable
def norm():
    return Normal(tensor(0.0), tensor(1.0))


@bm.functional
def f1by2():
    # a 1x2 tensor in Python becomes a 2x1 matrix in BMG
    return tensor([norm().exp(), norm()])


@bm.functional
def f2by1():
    # a 2x1 tensor in Python becomes a 1x2 matrix in BMG
    return tensor([[norm().exp()], [norm()]])


@bm.functional
def f2by3():
    # a 2x3 tensor in Python becomes a 3x2 matrix in BMG
    return tensor([[norm().exp(), 10, 20], [norm(), 30, 40]])


@bm.functional
def f1by2by3():
    # A 1x2x3 tensor in Python is an error in BMG.
    return tensor([[[norm().exp(), 10, 20], [norm(), 30, 40]]])


class ToMatrixTest(unittest.TestCase):
    def test_to_matrix_1by2(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f1by2()], {})
        observed = to_dot(
            bmg,
            node_types=True,
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
  N4[label="2:N"];
  N5[label="1:N"];
  N6[label="Exp:R+"];
  N7[label="ToReal:R"];
  N8[label="ToMatrix:MR[2,1]"];
  N9[label="Query:MR[2,1]"];
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
uint n4 = g.add_constant(2);
uint n5 = g.add_constant(1);
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
n0 = g.add_constant_real(0.0)
n1 = g.add_constant_pos_real(1.0)
n2 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n0, n1],
)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_constant_natural(2)
n5 = g.add_constant_natural(1)
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
  N4[label="2"];
  N5[label="1"];
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

    def test_to_matrix_2by1(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f2by1()], {})
        observed = to_dot(
            bmg,
            node_types=True,
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
n0 = g.add_constant_real(0.0)
n1 = g.add_constant_pos_real(1.0)
n2 = g.add_distribution(
  graph.DistributionType.NORMAL,
  graph.AtomicType.REAL,
  [n0, n1],
)
n3 = g.add_operator(graph.OperatorType.SAMPLE, [n2])
n4 = g.add_constant_natural(1)
n5 = g.add_constant_natural(2)
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

    def test_to_matrix_2by3(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f2by3()], {})
        observed = to_dot(
            bmg,
            node_types=True,
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
  N04[label="3:N"];
  N05[label="2:N"];
  N06[label="Exp:R+"];
  N07[label="ToReal:R"];
  N08[label="10.0:R"];
  N09[label="20.0:R"];
  N10[label="30.0:R"];
  N11[label="40.0:R"];
  N12[label="ToMatrix:MR[3,2]"];
  N13[label="Query:MR[3,2]"];
  N00 -> N02[label="mu:R"];
  N01 -> N02[label="sigma:R+"];
  N02 -> N03[label="operand:R"];
  N03 -> N06[label="operand:R"];
  N03 -> N12[label="3:R"];
  N04 -> N12[label="rows:N"];
  N05 -> N12[label="columns:N"];
  N06 -> N07[label="operand:<=R"];
  N07 -> N12[label="0:R"];
  N08 -> N12[label="1:R"];
  N09 -> N12[label="2:R"];
  N10 -> N12[label="4:R"];
  N11 -> N12[label="5:R"];
  N12 -> N13[label="operator:any"];
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
        bmg.add_query(lse0, _rv_id())
        bmg.add_query(lse1, _rv_id())

        observed = to_dot(
            bmg,
            node_types=True,
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

    def test_to_matrix_1by2by3(self) -> None:
        self.maxDiff = None
        bmg = BMGRuntime().accumulate_graph([f1by2by3()], {})

        # TODO: Error message could be more specific here than "a tensor".
        # We could say what is wrong: its size.

        expected = """
The model uses a tensor operation unsupported by Bean Machine Graph.
The unsupported node was created in function call f1by2by3()."""
        with self.assertRaises(ValueError) as ex:
            to_dot(
                bmg,
                node_types=True,
                edge_requirements=True,
                after_transform=True,
                label_edges=True,
            )
        self.assertEqual(expected.strip(), str(ex.exception).strip())
