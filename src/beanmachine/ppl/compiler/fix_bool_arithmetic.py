# Copyright (c) Facebook, Inc. and its affiliates.

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
            return self._typer.is_bool(n.left) or self._typer.is_bool(  # pyre-ignore
                n.right
            )
        # We can simplify 0+anything.
        if isinstance(n, bn.AdditionNode):
            return bn.is_zero(n.left) or bn.is_zero(n.right)

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
        if bn.is_zero(n.left):
            return n.left
        if bn.is_one(n.left):
            return n.right
        if bn.is_zero(n.right):
            return n.right
        if bn.is_one(n.right):
            return n.left
        zero = self._bmg.add_constant(0.0)
        if self._typer.is_bool(n.left):  # pyre-ignore
            return self._bmg.add_if_then_else(n.left, n.right, zero)
        assert self._typer.is_bool(n.right)
        return self._bmg.add_if_then_else(n.right, n.left, zero)

    def _fix_addition(self, n: bn.AdditionNode) -> bn.BMGNode:
        if bn.is_zero(n.left):
            return n.right
        assert bn.is_zero(n.right)
        return n.left

    def _fix_power(self, n: bn.PowerNode) -> bn.BMGNode:
        # x ** b  -->   if b then x else 1
        assert self._typer.is_bool(n.right)  # pyre-ignore
        one = self._bmg.add_constant(1.0)
        return self._bmg.add_if_then_else(n.right, n.left, one)
