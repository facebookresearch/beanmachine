# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import Size


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


class FixMatrixOpTest(unittest.TestCase):
    def test_fix_matrix_addition(self) -> None:
        bmg = BMGraphBuilder()
        zeros = bmg.add_real_matrix(torch.zeros(2))
        ones = bmg.add_pos_real_matrix(torch.ones(2))
        tensor_elements = []
        for index in range(0, 2):
            index_node = bmg.add_natural(index)
            index_mu = bmg.add_vector_index(zeros, index_node)
            index_sigma = bmg.add_vector_index(ones, index_node)
            normal = bmg.add_normal(index_mu, index_sigma)
            sample = bmg.add_sample(normal)
            tensor_elements.append(sample)
        matrix = bmg.add_tensor(Size([2]), *tensor_elements)
        exp = bmg.add_matrix_exp(matrix)
        mult = bmg.add_elementwise_multiplication(matrix, matrix)
        add = bmg.add_matrix_addition(exp, mult)
        bmg.add_query(add, _rv_id())
        observed = to_dot(bmg, after_transform=False)
        expectation = """
digraph "graph" {
  N00[label="[0.0,0.0]"];
  N01[label=0];
  N02[label=index];
  N03[label="[1.0,1.0]"];
  N04[label=index];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=index];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=Tensor];
  N13[label=MatrixExp];
  N14[label=ElementwiseMult];
  N15[label=MatrixAdd];
  N16[label=Query];
  N00 -> N02[label=left];
  N00 -> N08[label=left];
  N01 -> N02[label=right];
  N01 -> N04[label=right];
  N02 -> N05[label=mu];
  N03 -> N04[label=left];
  N03 -> N09[label=left];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N12[label=left];
  N07 -> N08[label=right];
  N07 -> N09[label=right];
  N08 -> N10[label=mu];
  N09 -> N10[label=sigma];
  N10 -> N11[label=operand];
  N11 -> N12[label=right];
  N12 -> N13[label=operand];
  N12 -> N14[label=left];
  N12 -> N14[label=right];
  N13 -> N15[label=left];
  N14 -> N15[label=right];
  N15 -> N16[label=operator];
}
                """
        self.assertEqual(expectation.strip(), observed.strip())
        observed = to_dot(bmg, after_transform=True)
        expectation = """
digraph "graph" {
  N00[label="[0.0,0.0]"];
  N01[label=0];
  N02[label=index];
  N03[label="[1.0,1.0]"];
  N04[label=index];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=index];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=2];
  N13[label=ToMatrix];
  N14[label=MatrixExp];
  N15[label=ToRealMatrix];
  N16[label=ElementwiseMult];
  N17[label=MatrixAdd];
  N18[label=Query];
  N00 -> N02[label=left];
  N00 -> N08[label=left];
  N01 -> N02[label=right];
  N01 -> N04[label=right];
  N02 -> N05[label=mu];
  N03 -> N04[label=left];
  N03 -> N09[label=left];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N13[label=0];
  N07 -> N08[label=right];
  N07 -> N09[label=right];
  N07 -> N13[label=columns];
  N08 -> N10[label=mu];
  N09 -> N10[label=sigma];
  N10 -> N11[label=operand];
  N11 -> N13[label=1];
  N12 -> N13[label=rows];
  N13 -> N14[label=operand];
  N13 -> N16[label=left];
  N13 -> N16[label=right];
  N14 -> N15[label=operand];
  N15 -> N17[label=left];
  N16 -> N17[label=right];
  N17 -> N18[label=operator];
}
        """
        self.assertEqual(expectation.strip(), observed.strip())

        generated_graph = to_bmg_graph(bmg)
        observed = generated_graph.graph.to_dot()
        expectation = """
digraph "graph" {
  N0[label="matrix"];
  N1[label="0"];
  N2[label="Index"];
  N3[label="matrix"];
  N4[label="Index"];
  N5[label="Normal"];
  N6[label="~"];
  N7[label="1"];
  N8[label="Index"];
  N9[label="Index"];
  N10[label="Normal"];
  N11[label="~"];
  N12[label="2"];
  N13[label="ToMatrix"];
  N14[label="Operator"];
  N15[label="ToReal"];
  N16[label="ElementwiseMultiply"];
  N17[label="MatrixAdd"];
  N0 -> N2;
  N0 -> N8;
  N1 -> N2;
  N1 -> N4;
  N2 -> N5;
  N3 -> N4;
  N3 -> N9;
  N4 -> N5;
  N5 -> N6;
  N6 -> N13;
  N7 -> N8;
  N7 -> N9;
  N7 -> N13;
  N8 -> N10;
  N9 -> N10;
  N10 -> N11;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N13 -> N16;
  N13 -> N16;
  N14 -> N15;
  N15 -> N17;
  N16 -> N17;
  Q0[label="Query"];
  N17 -> Q0;
}
        """
        self.assertEqual(expectation.strip(), observed.strip())

    def test_fix_elementwise_multiply(self) -> None:
        bmg = BMGraphBuilder()
        zeros = bmg.add_real_matrix(torch.zeros(2))
        ones = bmg.add_pos_real_matrix(torch.ones(2))
        tensor_elements = []
        for index in range(0, 2):
            index_node = bmg.add_natural(index)
            index_mu = bmg.add_vector_index(zeros, index_node)
            index_sigma = bmg.add_vector_index(ones, index_node)
            normal = bmg.add_normal(index_mu, index_sigma)
            sample = bmg.add_sample(normal)
            tensor_elements.append(sample)
        matrix = bmg.add_tensor(Size([2]), *tensor_elements)
        exp = bmg.add_matrix_exp(matrix)
        add = bmg.add_matrix_addition(matrix, matrix)
        mult = bmg.add_elementwise_multiplication(exp, add)
        sum = bmg.add_matrix_sum(mult)
        bmg.add_query(sum, _rv_id())
        observed = to_dot(bmg, after_transform=False)
        expectation = """
digraph "graph" {
  N00[label="[0.0,0.0]"];
  N01[label=0];
  N02[label=index];
  N03[label="[1.0,1.0]"];
  N04[label=index];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=index];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=Tensor];
  N13[label=MatrixExp];
  N14[label=MatrixAdd];
  N15[label=ElementwiseMult];
  N16[label=MatrixSum];
  N17[label=Query];
  N00 -> N02[label=left];
  N00 -> N08[label=left];
  N01 -> N02[label=right];
  N01 -> N04[label=right];
  N02 -> N05[label=mu];
  N03 -> N04[label=left];
  N03 -> N09[label=left];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N12[label=left];
  N07 -> N08[label=right];
  N07 -> N09[label=right];
  N08 -> N10[label=mu];
  N09 -> N10[label=sigma];
  N10 -> N11[label=operand];
  N11 -> N12[label=right];
  N12 -> N13[label=operand];
  N12 -> N14[label=left];
  N12 -> N14[label=right];
  N13 -> N15[label=left];
  N14 -> N15[label=right];
  N15 -> N16[label=operand];
  N16 -> N17[label=operator];
}
                """
        self.assertEqual(expectation.strip(), observed.strip())
        observed = to_dot(bmg, after_transform=True)
        expectation = """
digraph "graph" {
  N00[label="[0.0,0.0]"];
  N01[label=0];
  N02[label=index];
  N03[label="[1.0,1.0]"];
  N04[label=index];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=index];
  N10[label=Normal];
  N11[label=Sample];
  N12[label=2];
  N13[label=ToMatrix];
  N14[label=MatrixExp];
  N15[label=ToRealMatrix];
  N16[label=MatrixAdd];
  N17[label=ElementwiseMult];
  N18[label=MatrixSum];
  N19[label=Query];
  N00 -> N02[label=left];
  N00 -> N08[label=left];
  N01 -> N02[label=right];
  N01 -> N04[label=right];
  N02 -> N05[label=mu];
  N03 -> N04[label=left];
  N03 -> N09[label=left];
  N04 -> N05[label=sigma];
  N05 -> N06[label=operand];
  N06 -> N13[label=0];
  N07 -> N08[label=right];
  N07 -> N09[label=right];
  N07 -> N13[label=columns];
  N08 -> N10[label=mu];
  N09 -> N10[label=sigma];
  N10 -> N11[label=operand];
  N11 -> N13[label=1];
  N12 -> N13[label=rows];
  N13 -> N14[label=operand];
  N13 -> N16[label=left];
  N13 -> N16[label=right];
  N14 -> N15[label=operand];
  N15 -> N17[label=left];
  N16 -> N17[label=right];
  N17 -> N18[label=operand];
  N18 -> N19[label=operator];
}
        """
        self.assertEqual(expectation.strip(), observed.strip())

        generated_graph = to_bmg_graph(bmg)
        observed = generated_graph.graph.to_dot()
        expectation = """
digraph "graph" {
  N0[label="matrix"];
  N1[label="0"];
  N2[label="Index"];
  N3[label="matrix"];
  N4[label="Index"];
  N5[label="Normal"];
  N6[label="~"];
  N7[label="1"];
  N8[label="Index"];
  N9[label="Index"];
  N10[label="Normal"];
  N11[label="~"];
  N12[label="2"];
  N13[label="ToMatrix"];
  N14[label="Operator"];
  N15[label="ToReal"];
  N16[label="MatrixAdd"];
  N17[label="ElementwiseMultiply"];
  N18[label="Operator"];
  N0 -> N2;
  N0 -> N8;
  N1 -> N2;
  N1 -> N4;
  N2 -> N5;
  N3 -> N4;
  N3 -> N9;
  N4 -> N5;
  N5 -> N6;
  N6 -> N13;
  N7 -> N8;
  N7 -> N9;
  N7 -> N13;
  N8 -> N10;
  N9 -> N10;
  N10 -> N11;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
  N13 -> N16;
  N13 -> N16;
  N14 -> N15;
  N15 -> N17;
  N16 -> N17;
  N17 -> N18;
  Q0[label="Query"];
  N18 -> Q0;
}
        """
        self.assertEqual(expectation.strip(), observed.strip())

    def test_fix_matrix_sum(self) -> None:
        bmg = BMGraphBuilder()
        probs = bmg.add_real_matrix(torch.tensor([[0.75, 0.25], [0.125, 0.875]]))
        tensor_elements = []
        for row in range(0, 2):
            row_node = bmg.add_natural(row)
            row_prob = bmg.add_column_index(probs, row_node)
            for column in range(0, 2):
                col_index = bmg.add_natural(column)
                prob = bmg.add_vector_index(row_prob, col_index)
                bernoulli = bmg.add_bernoulli(prob)
                sample = bmg.add_sample(bernoulli)
                tensor_elements.append(sample)
        matrix = bmg.add_tensor(Size([2, 2]), *tensor_elements)
        sum = bmg.add_matrix_sum(matrix)
        bmg.add_query(sum, _rv_id())
        observed_beanstalk = to_dot(bmg, after_transform=True)
        expected = """
digraph "graph" {
  N00[label="[[0.75,0.25],\\\\n[0.125,0.875]]"];
  N01[label=0];
  N02[label=ColumnIndex];
  N03[label=index];
  N04[label=ToProb];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=ToProb];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=ColumnIndex];
  N13[label=index];
  N14[label=ToProb];
  N15[label=Bernoulli];
  N16[label=Sample];
  N17[label=index];
  N18[label=ToProb];
  N19[label=Bernoulli];
  N20[label=Sample];
  N21[label=2];
  N22[label=ToMatrix];
  N23[label=ToRealMatrix];
  N24[label=MatrixSum];
  N25[label=Query];
  N00 -> N02[label=left];
  N00 -> N12[label=left];
  N01 -> N02[label=right];
  N01 -> N03[label=right];
  N01 -> N13[label=right];
  N02 -> N03[label=left];
  N02 -> N08[label=left];
  N03 -> N04[label=operand];
  N04 -> N05[label=probability];
  N05 -> N06[label=operand];
  N06 -> N22[label=0];
  N07 -> N08[label=right];
  N07 -> N12[label=right];
  N07 -> N17[label=right];
  N08 -> N09[label=operand];
  N09 -> N10[label=probability];
  N10 -> N11[label=operand];
  N11 -> N22[label=1];
  N12 -> N13[label=left];
  N12 -> N17[label=left];
  N13 -> N14[label=operand];
  N14 -> N15[label=probability];
  N15 -> N16[label=operand];
  N16 -> N22[label=2];
  N17 -> N18[label=operand];
  N18 -> N19[label=probability];
  N19 -> N20[label=operand];
  N20 -> N22[label=3];
  N21 -> N22[label=columns];
  N21 -> N22[label=rows];
  N22 -> N23[label=operand];
  N23 -> N24[label=operand];
  N24 -> N25[label=operator];
}
        """

        self.assertEqual(observed_beanstalk.strip(), expected.strip())

        generated_graph = to_bmg_graph(bmg)
        observed_bmg = generated_graph.graph.to_dot()
        expectation = """
digraph "graph" {
  N0[label="matrix"];
  N1[label="0"];
  N2[label="ColumnIndex"];
  N3[label="Index"];
  N4[label="ToProb"];
  N5[label="Bernoulli"];
  N6[label="~"];
  N7[label="1"];
  N8[label="Index"];
  N9[label="ToProb"];
  N10[label="Bernoulli"];
  N11[label="~"];
  N12[label="ColumnIndex"];
  N13[label="Index"];
  N14[label="ToProb"];
  N15[label="Bernoulli"];
  N16[label="~"];
  N17[label="Index"];
  N18[label="ToProb"];
  N19[label="Bernoulli"];
  N20[label="~"];
  N21[label="2"];
  N22[label="ToMatrix"];
  N23[label="ToReal"];
  N24[label="Operator"];
  N0 -> N2;
  N0 -> N12;
  N1 -> N2;
  N1 -> N3;
  N1 -> N13;
  N2 -> N3;
  N2 -> N8;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
  N6 -> N22;
  N7 -> N8;
  N7 -> N12;
  N7 -> N17;
  N8 -> N9;
  N9 -> N10;
  N10 -> N11;
  N11 -> N22;
  N12 -> N13;
  N12 -> N17;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N22;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N22;
  N21 -> N22;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  Q0[label="Query"];
  N24 -> Q0;
}
"""
        self.assertEqual(expectation.strip(), observed_bmg.strip())

    def test_fix_matrix_exp(self) -> None:
        bmg = BMGraphBuilder()
        probs = bmg.add_real_matrix(torch.tensor([[0.75, 0.25], [0.125, 0.875]]))
        tensor_elements = []
        for row in range(0, 2):
            row_node = bmg.add_natural(row)
            row_prob = bmg.add_column_index(probs, row_node)
            for column in range(0, 2):
                col_index = bmg.add_natural(column)
                prob = bmg.add_vector_index(row_prob, col_index)
                bernoulli = bmg.add_bernoulli(prob)
                sample = bmg.add_sample(bernoulli)
                tensor_elements.append(sample)
        matrix = bmg.add_tensor(Size([2, 2]), *tensor_elements)
        sum = bmg.add_matrix_exp(matrix)
        bmg.add_query(sum, _rv_id())
        observed_beanstalk = to_dot(bmg, after_transform=True)
        expectation = """
digraph "graph" {
  N00[label="[[0.75,0.25],\\\\n[0.125,0.875]]"];
  N01[label=0];
  N02[label=ColumnIndex];
  N03[label=index];
  N04[label=ToProb];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=1];
  N08[label=index];
  N09[label=ToProb];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=ColumnIndex];
  N13[label=index];
  N14[label=ToProb];
  N15[label=Bernoulli];
  N16[label=Sample];
  N17[label=index];
  N18[label=ToProb];
  N19[label=Bernoulli];
  N20[label=Sample];
  N21[label=2];
  N22[label=ToMatrix];
  N23[label=ToRealMatrix];
  N24[label=MatrixExp];
  N25[label=Query];
  N00 -> N02[label=left];
  N00 -> N12[label=left];
  N01 -> N02[label=right];
  N01 -> N03[label=right];
  N01 -> N13[label=right];
  N02 -> N03[label=left];
  N02 -> N08[label=left];
  N03 -> N04[label=operand];
  N04 -> N05[label=probability];
  N05 -> N06[label=operand];
  N06 -> N22[label=0];
  N07 -> N08[label=right];
  N07 -> N12[label=right];
  N07 -> N17[label=right];
  N08 -> N09[label=operand];
  N09 -> N10[label=probability];
  N10 -> N11[label=operand];
  N11 -> N22[label=1];
  N12 -> N13[label=left];
  N12 -> N17[label=left];
  N13 -> N14[label=operand];
  N14 -> N15[label=probability];
  N15 -> N16[label=operand];
  N16 -> N22[label=2];
  N17 -> N18[label=operand];
  N18 -> N19[label=probability];
  N19 -> N20[label=operand];
  N20 -> N22[label=3];
  N21 -> N22[label=columns];
  N21 -> N22[label=rows];
  N22 -> N23[label=operand];
  N23 -> N24[label=operand];
  N24 -> N25[label=operator];
}
        """
        self.assertEqual(expectation.strip(), observed_beanstalk.strip())

        generated_graph = to_bmg_graph(bmg)
        observed_bmg = generated_graph.graph.to_dot()
        expectation = """
digraph "graph" {
  N0[label="matrix"];
  N1[label="0"];
  N2[label="ColumnIndex"];
  N3[label="Index"];
  N4[label="ToProb"];
  N5[label="Bernoulli"];
  N6[label="~"];
  N7[label="1"];
  N8[label="Index"];
  N9[label="ToProb"];
  N10[label="Bernoulli"];
  N11[label="~"];
  N12[label="ColumnIndex"];
  N13[label="Index"];
  N14[label="ToProb"];
  N15[label="Bernoulli"];
  N16[label="~"];
  N17[label="Index"];
  N18[label="ToProb"];
  N19[label="Bernoulli"];
  N20[label="~"];
  N21[label="2"];
  N22[label="ToMatrix"];
  N23[label="ToReal"];
  N24[label="Operator"];
  N0 -> N2;
  N0 -> N12;
  N1 -> N2;
  N1 -> N3;
  N1 -> N13;
  N2 -> N3;
  N2 -> N8;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
  N6 -> N22;
  N7 -> N8;
  N7 -> N12;
  N7 -> N17;
  N8 -> N9;
  N9 -> N10;
  N10 -> N11;
  N11 -> N22;
  N12 -> N13;
  N12 -> N17;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N22;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N22;
  N21 -> N22;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  Q0[label="Query"];
  N24 -> Q0;
}
"""
        self.assertEqual(expectation.strip(), observed_bmg.strip())
