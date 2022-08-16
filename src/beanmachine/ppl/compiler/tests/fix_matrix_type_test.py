# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.gen_dot import to_dot
from torch import Tensor, Size

class FixMatrixAdditionTest(unittest.TestCase):
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
        query = bmg.add_query(add)
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
  N13[label=UNKNOWN];
  N14[label=UNKNOWN];
  N15[label=UNKNOWN];
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
  N12 -> N13[label=UNKNOWN];
  N12 -> N14[label=UNKNOWN];
  N12 -> N14[label=UNKNOWN];
  N13 -> N15[label=UNKNOWN];
  N14 -> N15[label=UNKNOWN];
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
  N14[label=UNKNOWN];
  N15[label=ToRealMatrix];
  N16[label=UNKNOWN];
  N17[label=UNKNOWN];
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
  N13 -> N14[label=UNKNOWN];
  N13 -> N16[label=UNKNOWN];
  N13 -> N16[label=UNKNOWN];
  N14 -> N15[label=operand];
  N15 -> N17[label=UNKNOWN];
  N16 -> N17[label=UNKNOWN];
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