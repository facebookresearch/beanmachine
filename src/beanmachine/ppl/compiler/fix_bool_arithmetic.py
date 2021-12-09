# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class BoolArithmeticFixer(ProblemFixerBase):
    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        assert isinstance(self._typer, LatticeTyper)
        # We can simplify 1*anything, 0*anything or bool*anything
        # to anything, 0, or an if-then-else respectively.
        if isinstance(n, bn.MultiplicationNode):
            assert len(n.inputs) == 2
            return self._typer.is_bool(  # pyre-ignore
                n.inputs[0]
            ) or self._typer.is_bool(n.inputs[1])
        # We can simplify 0+anything.
        if isinstance(n, bn.AdditionNode):
            assert len(n.inputs) == 2
            return bn.is_zero(n.inputs[0]) or bn.is_zero(n.inputs[1])

        # We can simplify anything**bool.
        if isinstance(n, bn.PowerNode):
            return self._typer.is_bool(n.right)

        # TODO: We could support b ** n where b is bool and n is a natural
        # constant. If n is the constant zero then the result is just
        # the Boolean constant true; if n is a non-zero constant then
        # b ** n is simply b.
        #
        # NOTE: We CANNOT support b ** n where b is bool and n is a
        # non-constant natural and the result is bool. That would
        # have the semantics of "if b then true else n == 0" but we do
        # not have an equality operator on naturals in BMG.

        return False

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        if isinstance(n, bn.MultiplicationNode):
            return self._fix_multiplication(n)
        if isinstance(n, bn.AdditionNode):
            return self._fix_addition(n)
        assert isinstance(n, bn.PowerNode)
        return self._fix_power(n)

    def _fix_multiplication(self, n: bn.MultiplicationNode) -> bn.BMGNode:
        assert len(n.inputs) == 2
        if bn.is_zero(n.inputs[0]):
            return n.inputs[0]
        if bn.is_one(n.inputs[0]):
            return n.inputs[1]
        if bn.is_zero(n.inputs[1]):
            return n.inputs[1]
        if bn.is_one(n.inputs[1]):
            return n.inputs[0]
        zero = self._bmg.add_constant(0.0)
        if self._typer.is_bool(n.inputs[0]):  # pyre-ignore
            return self._bmg.add_if_then_else(n.inputs[0], n.inputs[1], zero)
        assert self._typer.is_bool(n.inputs[1])
        return self._bmg.add_if_then_else(n.inputs[1], n.inputs[0], zero)

    def _fix_addition(self, n: bn.AdditionNode) -> bn.BMGNode:
        assert len(n.inputs) == 2
        if bn.is_zero(n.inputs[0]):
            return n.inputs[1]
        assert bn.is_zero(n.inputs[1])
        return n.inputs[0]

    def _fix_power(self, n: bn.PowerNode) -> bn.BMGNode:
        # x ** b  -->   if b then x else 1
        assert self._typer.is_bool(n.right)  # pyre-ignore
        one = self._bmg.add_constant(1.0)
        return self._bmg.add_if_then_else(n.right, n.left, one)
