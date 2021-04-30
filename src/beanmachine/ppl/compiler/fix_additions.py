# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


class AdditionFixer(ProblemFixerBase):
    """This class takes a Bean Machine Graph builder and attempts to
    rewrite reachable additions of the form:

    * add(1, negate(prob)) or add(negate(prob), 1) -> complement(prob)
    * add(1, negate(bool)) or add(negate(bool), 1) -> complement(bool)"""

    def __init__(self, bmg: BMGraphBuilder, typer: LatticeTyper) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)

    def _can_be_complement(self, n: bn.AdditionNode) -> bool:
        if bn.is_one(n.left):
            other = n.right
            if isinstance(other, bn.NegateNode):
                it = other.operand.inf_type
                if bt.supremum(it, bt.Probability) == bt.Probability:
                    return True
        if bn.is_one(n.right):
            other = n.left
            if isinstance(other, bn.NegateNode):
                it = other.operand.inf_type
                if bt.supremum(it, bt.Probability) == bt.Probability:
                    return True
        return False

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return isinstance(n, bn.AdditionNode) and self._can_be_complement(n)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        assert isinstance(n, bn.AdditionNode)
        assert self._can_be_complement(n)
        # We have 1+(-x) or (-x)+1 where x is either P or B, and require
        # a P or B. Complement(x) is of the same type as x if x is P or B.
        if bn.is_one(n.left):
            other = n.right
        else:
            assert bn.is_one(n.right)
            other = n.left
        assert isinstance(other, bn.NegateNode)
        return self._bmg.add_complement(other.operand)
