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


def addition_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    """This fixer rewrites additions into complements:
    * add(1, negate(prob)) or add(negate(prob), 1) -> complement(prob)
    * add(1, negate(bool)) or add(negate(bool), 1) -> complement(bool)"""

    def fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.AdditionNode) or len(node.inputs) != 2:
            return Inapplicable
        left = node.inputs[0]
        right = node.inputs[1]
        if (
            bn.is_one(left)
            and isinstance(right, bn.NegateNode)
            and typer.is_prob_or_bool(right.operand)
        ):
            return bmg.add_complement(right.operand)
        if (
            bn.is_one(right)
            and isinstance(left, bn.NegateNode)
            and typer.is_prob_or_bool(left.operand)
        ):
            return bmg.add_complement(left.operand)
        return Inapplicable

    return fixer
