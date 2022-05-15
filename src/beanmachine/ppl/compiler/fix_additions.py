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


def sum_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    """This fixer rewrites vector sums into multiary additions."""

    def fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.SumNode):
            return Inapplicable
        t = typer[node.operand]
        if not isinstance(t, bt.BMGMatrixType):
            return Inapplicable

        # TODO: Write code to handle a 2-d tensor element sum.
        if t.columns != 1:
            return Inapplicable

        indices = []
        for i in range(t.rows):
            c = bmg.add_constant(i)
            index = bmg.add_index(node.operand, c)
            indices.append(index)

        if len(indices) == 1:
            return indices[0]
        if len(indices) == 2:
            return bmg.add_addition(indices[0], indices[1])
        return bmg.add_multi_addition(*indices)

    return fixer
