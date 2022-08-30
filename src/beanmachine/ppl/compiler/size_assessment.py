# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import BMGMatrixType, RealMatrix
from beanmachine.ppl.compiler.error_report import BadMatrixMultiplication, BMGError
from beanmachine.ppl.compiler.sizer import is_scalar, Size, Sizer


class SizeAssessment:
    def __init__(self, sizer: Sizer):
        self.sizer = sizer

    def size_error(
        self, node: bn.BMGNode, context: BMGraphBuilder
    ) -> Optional[BMGError]:
        error = None
        if isinstance(node, bn.MatrixMultiplicationNode):
            lhs = node.inputs.inputs[0]
            rhs = node.inputs.inputs[1]

            lhs_size = self.sizer[lhs]
            rhs_size = self.sizer[rhs]

            if not (is_scalar(lhs_size) or is_scalar(rhs_size)):
                l_rhs = len(rhs_size)
                l_lhs = len(lhs_size)
                rhs_can_be_considered_column = (
                    l_rhs == 1 and l_lhs == 2 and lhs_size[1] == rhs_size[0]
                )
                lhs_can_be_considered_row = (
                    l_lhs == 1 and l_rhs == 2 and lhs_size[0] == rhs_size[0]
                )
                can_be_inner_product = (
                    l_rhs == 1 and l_lhs == 1 and rhs_size[0] == lhs_size[0]
                )
                are_not_matrices_or_not_compatible_matrices = (
                    not (len(lhs_size) == 2 and l_rhs == 2)
                ) or (lhs_size[1] != rhs_size[0])
                if are_not_matrices_or_not_compatible_matrices and not (
                    rhs_can_be_considered_column
                    or lhs_can_be_considered_row
                    or can_be_inner_product
                ):
                    # Do NOT use the Lattice typer. the BMGMatrix type constructor
                    # will translate the size into column major form. Since the user is writing in
                    # a row major api, we present the error message in row major form. We don't care about the
                    # element type in this case
                    def get_matrix_type(sz: Size) -> BMGMatrixType:
                        rows = 1
                        cols = 1
                        length = len(sz)
                        if length >= 2:
                            rows = sz[length - 2]
                            cols = sz[length - 1]
                        elif length == 1:
                            rows = sz[0]
                        return RealMatrix(rows, cols)

                    error = BadMatrixMultiplication(
                        node,
                        get_matrix_type(lhs_size),
                        get_matrix_type(rhs_size),
                        context.execution_context.node_locations(node),
                    )

        return error
