# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import BadMatrixMultiplication
from beanmachine.ppl.compiler.size_assessment import SizeAssessment

from beanmachine.ppl.compiler.sizer import Size, Sizer


class SizeAssessmentTests(unittest.TestCase):
    def test_matrix_mult(self):
        bmg = BMGraphBuilder()
        assessor = SizeAssessment(Sizer())
        probs = bmg.add_real_matrix(
            torch.tensor([[0.5, 0.125, 0.125], [0.0625, 0.0625, 0.875]])
        )
        tensor_elements = []
        for row in range(0, 2):
            row_node = bmg.add_natural(row)
            row_prob = bmg.add_column_index(probs, row_node)
            for column in range(0, 3):
                col_index = bmg.add_natural(column)
                prob = bmg.add_vector_index(row_prob, col_index)
                bernoulli = bmg.add_bernoulli(prob)
                sample = bmg.add_sample(bernoulli)
                tensor_elements.append(sample)
        matrix2by3_rhs = bmg.add_tensor(Size([2, 3]), *tensor_elements)
        # invalid
        matrix2by3 = bmg.add_real_matrix(
            torch.tensor([[0.21, 0.27, 0.3], [0.5, 0.6, 0.1]])
        )
        matrix1by3 = bmg.add_real_matrix(torch.tensor([[0.1, 0.2, 0.3]]))
        matrix3 = bmg.add_real_matrix(torch.tensor([0.1, 0.2, 0.9]))
        scalar = bmg.add_real(4.5)

        mm_invalid = bmg.add_matrix_multiplication(matrix2by3_rhs, matrix2by3)
        error_size_mismatch = assessor.size_error(mm_invalid, bmg)
        self.assertIsInstance(error_size_mismatch, BadMatrixMultiplication)
        expectation = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The dimensions of the operands are 2x3 and 2x3.
        """
        self.assertEqual(expectation.strip(), error_size_mismatch.__str__().strip())

        broadcast_not_supported_yet = bmg.add_matrix_multiplication(
            matrix2by3_rhs, matrix1by3
        )
        error_broadcast_not_supported_yet = assessor.size_error(
            broadcast_not_supported_yet, bmg
        )
        expectation = """
The model uses a matrix multiplication (@) operation unsupported by Bean Machine Graph.
The dimensions of the operands are 2x3 and 1x3.
        """
        self.assertEqual(
            expectation.strip(), error_broadcast_not_supported_yet.__str__().strip()
        )
        errors = [
            assessor.size_error(bmg.add_matrix_multiplication(matrix2by3_rhs, mm), bmg)
            for mm in [matrix3, scalar]
        ]
        for error in errors:
            self.assertIsNone(error)
