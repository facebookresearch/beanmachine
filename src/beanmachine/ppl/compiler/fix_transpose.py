# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


def _is_real_matrix(t: bt.BMGLatticeType) -> bool:
    return any(
        isinstance(t, m)
        for m in {
            bt.RealMatrix,
            bt.PositiveRealMatrix,
            bt.NegativeRealMatrix,
            bt.ProbabilityMatrix,
        }
    )


def identity_transpose_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    """This fixer eliminates scalar or 1x1 transposes"""

    def fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.TransposeNode):
            return Inapplicable
        in_node = node.inputs[0]
        if not typer.is_matrix(in_node):
            return in_node
        return Inapplicable

    return fixer
