# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_cpp import to_bmg_cpp
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_bmg_python import to_bmg_python
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.model.rv_identifier import RVIdentifier


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


class ColumnIndexTest(unittest.TestCase):
    def test_column_index_1(self) -> None:

        self.maxDiff = None
        bmg = BMGraphBuilder()
        t = bmg.add_natural(2)
        o = bmg.add_natural(1)
        z = bmg.add_natural(0)
        n = bmg.add_normal(z, o)
        ns = bmg.add_sample(n)
        e = bmg.add_exp(ns)
        h = bmg.add_probability(0.5)
        b = bmg.add_bernoulli(h)
        bs = bmg.add_sample(b)
        m = bmg.add_to_matrix(t, t, e, ns, ns, ns)
        ci = bmg.add_column_index(m, bs)
        lsev = bmg.add_logsumexp_vector(ci)
        bmg.add_query(lsev, _rv_id())

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
  N04[label="0.5:P"];
  N05[label="Bernoulli:B"];
  N06[label="Sample:B"];
  N07[label="2:N"];
  N08[label="Exp:R+"];
  N09[label="ToReal:R"];
  N10[label="ToMatrix:MR[2,2]"];
  N11[label="1:N"];
  N12[label="0:N"];
  N13[label="if:N"];
  N14[label="ColumnIndex:MR[2,1]"];
  N15[label="LogSumExp:R"];
  N16[label="Query:R"];
  N00 -> N02[label="mu:R"];
  N01 -> N02[label="sigma:R+"];
  N02 -> N03[label="operand:R"];
  N03 -> N08[label="operand:R"];
  N03 -> N10[label="1:R"];
  N03 -> N10[label="2:R"];
  N03 -> N10[label="3:R"];
  N04 -> N05[label="probability:P"];
  N05 -> N06[label="operand:B"];
  N06 -> N13[label="condition:B"];
  N07 -> N10[label="columns:N"];
  N07 -> N10[label="rows:N"];
  N08 -> N09[label="operand:<=R"];
  N09 -> N10[label="0:R"];
  N10 -> N14[label="left:MR[2,2]"];
  N11 -> N13[label="consequence:N"];
  N12 -> N13[label="alternative:N"];
  N13 -> N14[label="right:N"];
  N14 -> N15[label="operand:MR[2,1]"];
  N15 -> N16[label="operator:any"];
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
uint n7 = g.add_constant(2);
uint n8 = g.add_operator(
  graph::OperatorType::EXP, std::vector<uint>({n3}));
uint n9 = g.add_operator(
  graph::OperatorType::TO_REAL, std::vector<uint>({n8}));
uint n10 = g.add_operator(
  graph::OperatorType::TO_MATRIX,
  std::vector<uint>({n7, n7, n9, n3, n3, n3}));
uint n11 = g.add_constant(1);
uint n12 = g.add_constant(0);
uint n13 = g.add_operator(
  graph::OperatorType::IF_THEN_ELSE,
  std::vector<uint>({n6, n11, n12}));
uint n14 = g.add_operator(
  graph::OperatorType::COLUMN_INDEX, std::vector<uint>({n10, n13}));
uint n15 = g.add_operator(
  graph::OperatorType::LOGSUMEXP_VECTOR, std::vector<uint>({n14}));
uint q0 = g.query(n15);
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
n4 = g.add_constant_probability(0.5)
n5 = g.add_distribution(
  graph.DistributionType.BERNOULLI,
  graph.AtomicType.BOOLEAN,
  [n4],
)
n6 = g.add_operator(graph.OperatorType.SAMPLE, [n5])
n7 = g.add_constant_natural(2)
n8 = g.add_operator(graph.OperatorType.EXP, [n3])
n9 = g.add_operator(graph.OperatorType.TO_REAL, [n8])
n10 = g.add_operator(
  graph.OperatorType.TO_MATRIX,
  [n7, n7, n9, n3, n3, n3],
)
n11 = g.add_constant_natural(1)
n12 = g.add_constant_natural(0)
n13 = g.add_operator(
  graph.OperatorType.IF_THEN_ELSE,
  [n6, n11, n12],
)
n14 = g.add_operator(graph.OperatorType.COLUMN_INDEX, [n10, n13])
n15 = g.add_operator(graph.OperatorType.LOGSUMEXP_VECTOR, [n14])
q0 = g.query(n15)
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
  N7[label="2"];
  N8[label="exp"];
  N9[label="ToReal"];
  N10[label="ToMatrix"];
  N11[label="1"];
  N12[label="0"];
  N13[label="IfThenElse"];
  N14[label="ColumnIndex"];
  N15[label="LogSumExp"];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N8;
  N3 -> N10;
  N3 -> N10;
  N3 -> N10;
  N4 -> N5;
  N5 -> N6;
  N6 -> N13;
  N7 -> N10;
  N7 -> N10;
  N8 -> N9;
  N9 -> N10;
  N10 -> N14;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  Q0[label="Query"];
  N15 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
