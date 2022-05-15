# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    node_fixer_first_match,
    NodeFixer,
    NodeFixerResult,
    type_guard,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class BoolArithmeticFixer:
    _bmg: BMGraphBuilder
    _typer: LatticeTyper

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        self._bmg = bmg
        self._typer = typer

    def _fix_multiplication(self, n: bn.MultiplicationNode) -> NodeFixerResult:
        # We can simplify 1*anything, 0*anything or bool*anything
        # to anything, 0, or an if-then-else respectively.

        # TODO: We could extend this to multiary multiplication.
        if len(n.inputs) != 2:
            return Inapplicable
        if bn.is_zero(n.inputs[0]):
            return n.inputs[0]
        if bn.is_one(n.inputs[0]):
            return n.inputs[1]
        if bn.is_zero(n.inputs[1]):
            return n.inputs[1]
        if bn.is_one(n.inputs[1]):
            return n.inputs[0]
        if self._typer.is_bool(n.inputs[0]):
            zero = self._bmg.add_constant(0.0)
            return self._bmg.add_if_then_else(n.inputs[0], n.inputs[1], zero)
        if self._typer.is_bool(n.inputs[1]):
            zero = self._bmg.add_constant(0.0)
            return self._bmg.add_if_then_else(n.inputs[1], n.inputs[0], zero)
        return Inapplicable

    def _fix_addition(self, n: bn.AdditionNode) -> NodeFixerResult:
        # We can simplify 0+anything.
        # TODO: We could extend this to multiary addition.
        if len(n.inputs) != 2:
            return Inapplicable
        if bn.is_zero(n.inputs[0]):
            return n.inputs[1]
        if bn.is_zero(n.inputs[1]):
            return n.inputs[0]
        return Inapplicable

    def _fix_power(self, n: bn.PowerNode) -> NodeFixerResult:
        # x ** b  -->   if b then x else 1
        if self._typer.is_bool(n.right):
            one = self._bmg.add_constant(1.0)
            return self._bmg.add_if_then_else(n.right, n.left, one)
        return Inapplicable


def bool_arithmetic_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    baf = BoolArithmeticFixer(bmg, typer)
    return node_fixer_first_match(
        [
            type_guard(bn.AdditionNode, baf._fix_addition),
            type_guard(bn.MultiplicationNode, baf._fix_multiplication),
            type_guard(bn.PowerNode, baf._fix_power),
        ]
    )
