# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class MatrixScaleFixer(ProblemFixerBase):
    """This class takes a Bean Machine Graph builder and attempts to
    rewrite binary multiplications that involve a matrix and a scalar into
    a matrix_scale node.
    """

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        # See note in fix_unsupported.py for why this is temporarily disabled.
        return False
        """
        # A matrix multiplication is fixable (to matrix_scale) if it is
        # a binary multiplication with non-singlton result type
        # and the type of one argument is matrix and the other is scalar
        if not isinstance(n, bn.MultiplicationNode) or not (len(n.inputs) == 2):
            return False
        # The return type of the node should be matrix
        if not (self._typer[n]).is_singleton():
            return False
        # Now let's check the types of the inputs
        input_types = [self._typer[i] for i in n.inputs]
        # If both are scalar, then there is nothing to do
        if all(t.is_singleton() for t in input_types):
            return False  # Both are scalar
        # If both are matrices, then there is nothing to do
        if all(not (t.is_singleton()) for t in input_types):
            return False  # Both are matrices
        return True"""

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # Double check we can proceed, and that exactly one is scalar
        assert self._needs_fixing(n)
        # We'll need to name the inputs and their types
        left, right = n.inputs
        left_type, right_type = [self._typer[i] for i in n.inputs]
        # Let's assume the first is the scalr one
        scalar, matrix = left, right
        # Fix it if necessary
        if right_type.is_singleton():
            scalar = right
            matrix = left
        return self._bmg.add_matrix_scale(scalar, matrix)
