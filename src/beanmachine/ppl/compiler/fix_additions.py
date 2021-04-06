# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import One
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase


class AdditionFixer(ProblemFixerBase):
    """This class takes a Bean Machine Graph builder and attempts to
    rewrite reachable additions of the form:

    * add(1, negate(prob)) or add(negate(prob), 1) -> complement(prob)
    * add(1, negate(bool)) or add(negate(bool), 1) -> complement(bool)"""

    def __init__(self, bmg: BMGraphBuilder) -> None:
        ProblemFixerBase.__init__(self, bmg)

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return isinstance(n, bn.AdditionNode) and n.can_be_complement

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        assert isinstance(n, bn.AdditionNode)
        assert n.can_be_complement
        # We have 1+(-x) or (-x)+1 where x is either P or B, and require
        # a P or B. Complement(x) is of the same type as x if x is P or B.
        if n.left.inf_type == One:
            other = n.right
        else:
            assert n.right.inf_type == One
            other = n.left
        assert isinstance(other, bn.NegateNode)
        return self._bmg.add_complement(other.operand)
