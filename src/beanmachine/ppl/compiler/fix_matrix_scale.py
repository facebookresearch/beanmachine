# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    NodeFixer,
    NodeFixerResult,
    Inapplicable,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper

"""This node fixer takes a Bean Machine Graph builder and attempts to
rewrite binary multiplications that involve a matrix and a scalar into
a matrix_scale node.
"""


def matrix_scale_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    def fixer(n: bn.BMGNode) -> NodeFixerResult:
        # A matrix multiplication is fixable (to matrix_scale) if it is
        # a binary multiplication with non-singleton result type
        # and the type of one argument is matrix and the other is scalar
        if not isinstance(n, bn.MultiplicationNode) or len(n.inputs) != 2:
            return Inapplicable
        # The return type of the node should be matrix
        if not typer[n].is_singleton():
            return Inapplicable
        # Now let's check the types of the inputs
        input_types = [typer[i] for i in n.inputs]
        # If both are scalar, then there is nothing to do
        if all(t.is_singleton() for t in input_types):
            return Inapplicable  # Both are scalar
        # If both are matrices, then there is nothing to do
        if all(not (t.is_singleton()) for t in input_types):
            return Inapplicable  # Both are matrices
        left, right = n.inputs
        if input_types[1].is_singleton():
            scalar, matrix = right, left
        else:
            scalar, matrix = left, right
        return bmg.add_matrix_scale(scalar, matrix)

    return fixer
