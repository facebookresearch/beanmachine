# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
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

    def _is_one_minus_prob(self, x: bn.BMGNode, y: bn.BMGNode) -> bool:
        return (
            bn.is_one(x)
            and isinstance(y, bn.NegateNode)
            and self._typer.is_prob_or_bool(y.operand)  # pyre-ignore
        )

    def _can_be_complement(self, n: bn.AdditionNode) -> bool:
        assert len(n.inputs) == 2
        return self._is_one_minus_prob(
            n.inputs[0], n.inputs[1]
        ) or self._is_one_minus_prob(n.inputs[1], n.inputs[0])

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        return isinstance(n, bn.AdditionNode) and self._can_be_complement(n)

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        assert isinstance(n, bn.AdditionNode)
        assert len(n.inputs) == 2
        assert self._can_be_complement(n)
        # We have 1+(-x) or (-x)+1 where x is either P or B, and require
        # a P or B. Complement(x) is of the same type as x if x is P or B.
        if bn.is_one(n.inputs[0]):
            other = n.inputs[1]
        else:
            assert bn.is_one(n.inputs[1])
            other = n.inputs[0]
        assert isinstance(other, bn.NegateNode)
        return self._bmg.add_complement(other.operand)
