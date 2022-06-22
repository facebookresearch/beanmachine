# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)


class MultiaryOperatorFixer:
    """This fixer transforms graphs with long chains of binary operator nodes
    into multiary operations. This greatly decreases both the number of nodes
    and the number of edges in the graph, which can lead to performance wins
    during inference."""

    _bmg: BMGraphBuilder
    _operator: type

    def __init__(self, bmg: BMGraphBuilder, operator: type) -> None:
        self._bmg = bmg
        self._operator = operator

    def _single_output_is_operator(self, n: bn.BMGNode) -> bool:
        if len(n.outputs.items) != 1:
            # Not exactly one output node.
            return False

        if next(iter(n.outputs.items.values())) != 1:
            # Exactly one output node, but has two edges going to it.
            # TODO: This is a bit opaque. Add a helper method for this.
            return False

        o = next(iter(n.outputs.items.keys()))
        return isinstance(o, self._operator)

    def _addition_single_output_is_operator(self, n: bn.BMGNode) -> bool:
        if not isinstance(n, self._operator):
            return False
        if len(n.inputs) > 2:
            return False
        return self._single_output_is_operator(n)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        # A binary operator is fixable if:
        #
        # * There is more than one output OR the single output is NOT the given operation
        # * At least one of the left or right inputs is a binary operator with only
        #   one output.
        #
        # Let us say the operator is addition, we are looking for stuff like:
        #
        #  A   B
        #   \ /
        #    +   C
        #     \ /
        #      +   D
        #       \ /
        #        +
        #
        # to turn it into
        #
        #  A  B C  D
        #   \ | | /
        #     sum
        #
        # Why do we have these conditions?
        #
        # * Consider the (A + B) + C node. We do not want to fix it.
        #   If there is exactly one output and it is an addition, then
        #   this node's output is itself a candidate for fixing; we can skip
        #   this one and fix it instead. No need to do extra work we're just
        #   going to throw away.
        #
        # * Any addition with two or more outputs is an addition that is
        #   deduplicated. We do not want to eliminate it; doing so causes
        #   the deduplicated work to be done twice. That is, if we have
        #
        #  A   B
        #   \ /
        #    +   C
        #     \ /
        #  E   +   D
        #   \ / \ /
        #    *   +
        #
        # Then the bottom addition node is NOT fixable but the A + B + C addition
        # is fixable. The desired final graph is:
        #
        #   A  B  C
        #    \ | /
        #  E  sum  D
        #   \ / \ /
        #    *   +
        #
        # and NOT
        #
        #   A  B  C
        #    \ | /
        #  E  sum    A  B C  D
        #   \ /       \ | | /
        #    *          sum
        #
        # Why not? Because our metrics are graph size and amount of arithmetic
        # performed when evaluating the graph in BMG.
        #
        # * The original graph has eight edges, nine nodes, and computes three additions:
        #   t1 = A + B, t2 = t1 + C, t3 = t2 + D
        # * The desired graph has seven edges, eight nodes, and computes three additions:
        #   t1 = sum(A, B, C) requires two additions, and t2 = t1 + D is one more.
        # The bad graph has nine edges, eight nodes, and computes five additions:
        # sum(A, B, C) does two additions and sum(A, B, C, D) does three.
        #
        # The desired graph is a clear win in its reduced edge and node count without
        # actually doing more math. The bad graph is in every way worse than the
        # desired graph.
        return (
            isinstance(n, self._operator)
            and len(n.inputs) == 2
            and not self._single_output_is_operator(n)
            and (
                self._addition_single_output_is_operator(n.inputs[0])
                or self._addition_single_output_is_operator(n.inputs[1])
            )
        )

    def accumulate_input_nodes(self, n: bn.BMGNode) -> List[bn.BMGNode]:
        acc = []
        stack = [n.inputs[1], n.inputs[0]]
        while len(stack) > 0:
            c = stack.pop()
            if self._addition_single_output_is_operator(c):
                assert isinstance(c, self._operator)
                assert len(n.inputs) == 2
                stack.append(c.inputs[1])
                stack.append(c.inputs[0])
            else:
                acc.append(c)
        return acc


def multiary_addition_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    """This fixer transforms graphs with long chains of binary addition nodes
    into multiary addition. This greatly decreases both the number of nodes
    and the number of edges in the graph, which can lead to performance wins
    during inference."""

    maf = MultiaryOperatorFixer(bmg, bn.AdditionNode)

    def multiary_addition_fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not maf._needs_fixing(node):
            return Inapplicable
        acc = maf.accumulate_input_nodes(node)
        return bmg.add_multi_addition(*acc)

    return multiary_addition_fixer


def multiary_multiplication_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    """This fixer transforms graphs with long chains of binary multiplication nodes
    into multiary multiplication. This greatly decreases both the number of nodes
    and the number of edges in the graph, which can lead to performance wins
    during inference."""
    maf = MultiaryOperatorFixer(bmg, bn.MultiplicationNode)

    def multiary_multiplication_fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not maf._needs_fixing(node):
            return Inapplicable
        acc = maf.accumulate_input_nodes(node)
        return bmg.add_multi_multiplication(*acc)

    return multiary_multiplication_fixer
