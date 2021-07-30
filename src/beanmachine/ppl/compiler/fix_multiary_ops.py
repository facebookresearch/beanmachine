# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.typer_base import TyperBase


class MultiaryAdditionFixer(ProblemFixerBase):
    """This fixer transforms graphs with long chains of binary addition nodes
    into multiary additions. This greatly decreases both the number of nodes
    and the number of edges in the graph, which can lead to performance wins
    during inference."""

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _single_output_is_addition(self, n: bn.BMGNode) -> bool:
        if len(n.outputs.items) != 1:
            # Not exactly one output node.
            return False

        if next(iter(n.outputs.items.values())) != 1:
            # Exactly one output node, but has two edges going to it.
            # TODO: This is a bit opaque. Add a helper method for this.
            return False

        o = next(iter(n.outputs.items.keys()))
        return isinstance(o, bn.AdditionNode)

    def _addition_single_output_is_addition(self, n: bn.BMGNode) -> bool:
        if not isinstance(n, bn.AdditionNode):
            return False
        if len(n.inputs) > 2:
            return False
        return self._single_output_is_addition(n)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        # A binary addition is fixable if:
        #
        # * There is more than one output OR the single output is NOT an addition
        # * At least one of the left or right inputs is a binary addition with only
        #   one output.
        #
        # That is, we are looking for stuff like:
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
            isinstance(n, bn.AdditionNode)
            and len(n.inputs) == 2
            and not self._single_output_is_addition(n)
            and (
                self._addition_single_output_is_addition(n.inputs[0])
                or self._addition_single_output_is_addition(n.inputs[1])
            )
        )

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # We require that this algorithm be non-recursive because the
        # path through the graph could be longer than the Python
        # recursion limit.
        assert isinstance(n, bn.AdditionNode)
        assert len(n.inputs) == 2
        acc = []
        stack = [n.inputs[1], n.inputs[0]]
        while len(stack) > 0:
            c = stack.pop()
            if self._addition_single_output_is_addition(c):
                assert isinstance(c, bn.AdditionNode)
                assert len(n.inputs) == 2
                stack.append(c.inputs[1])
                stack.append(c.inputs[0])
            else:
                acc.append(c)
        assert len(acc) >= 3
        return self._bmg.add_multi_addition(*acc)


class MultiaryMultiplicationFixer(ProblemFixerBase):
    """This fixer transforms graphs with long chains of binary multiplication nodes
    into multiary multiplication. This greatly decreases both the number of nodes
    and the number of edges in the graph, which can lead to performance wins
    during inference."""

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _single_output_is_multiplication(self, n: bn.BMGNode) -> bool:
        if len(n.outputs.items) != 1:
            # Not exactly one output node.
            return False

        if next(iter(n.outputs.items.values())) != 1:
            # Exactly one output node, but has two edges going to it.
            # TODO: This is a bit opaque. Add a helper method for this.
            return False

        o = next(iter(n.outputs.items.keys()))
        return isinstance(o, bn.MultiplicationNode)

    def _multiplication_single_output_is_multiplication(self, n: bn.BMGNode) -> bool:
        if not isinstance(n, bn.MultiplicationNode):
            return False
        if len(n.inputs) > 2:
            return False
        return self._single_output_is_multiplication(n)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        # This follows the same heuristic as multiary addition
        return (
            isinstance(n, bn.MultiplicationNode)
            and len(n.inputs) == 2
            and not self._single_output_is_multiplication(n)
            and (
                self._multiplication_single_output_is_multiplication(n.inputs[0])
                or self._multiplication_single_output_is_multiplication(n.inputs[1])
            )
        )

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        # We require that this algorithm be non-recursive because the
        # path through the graph could be longer than the Python
        # recursion limit.
        assert isinstance(n, bn.MultiplicationNode)
        assert len(n.inputs) == 2
        acc = []
        stack = [n.inputs[1], n.inputs[0]]
        while len(stack) > 0:
            c = stack.pop()
            if self._multiplication_single_output_is_multiplication(c):
                assert isinstance(c, bn.MultiplicationNode)
                assert len(n.inputs) == 2
                stack.append(c.inputs[1])
                stack.append(c.inputs[0])
            else:
                acc.append(c)
        assert len(acc) >= 3
        return self._bmg.add_multi_multiplication(*acc)
