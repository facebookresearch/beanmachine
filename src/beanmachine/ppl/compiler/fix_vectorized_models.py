# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_problem import ProblemFixerBase
from beanmachine.ppl.compiler.sizer import Sizer
from beanmachine.ppl.compiler.typer_base import TyperBase
from torch import Size

# This class turns vectorized models into unvectorized models.
#
# TODO: For now we only implement a proof of concept. The model:
#
# @rv def flip(): return Bernoulli(tensor([0.25, 0.75]))
#
# which we cannot represent in BMG is rewritten into the model:
#
# p = tensor([0.25, 0.75])
# @rv def f0: return Bernoulli(p[0])
# @rv def f1: return Bernoulli(p[1])
# @functional def flip(): return tensor([f0()), f1())])
#
# which we can represent in BMG.
#
# TODO: If we have an observation on flip() in the original model we need
# to rewrite it into observations of f0() and f1()


class VectorizedDistributionFixer(ProblemFixerBase):

    fixed_one: bool

    def __init__(self, bmg: BMGraphBuilder, typer: Sizer) -> None:
        ProblemFixerBase.__init__(self, bmg, typer)
        self.fixed_one = False

    def _needs_fixing(self, n: bn.BMGNode) -> bool:
        if not isinstance(n, bn.SampleNode):
            return False
        dist = n.operand
        if not isinstance(dist, bn.BernoulliNode):
            return False
        s = self._typer[dist.probability]
        return s == Size([2])

    def _add_sample(self, node: bn.BernoulliNode, i: int) -> bn.SampleNode:
        p = node.probability
        assert self._typer[p] == Size([2])
        ci = self._bmg.add_constant(i)
        pi = self._bmg.add_index(p, ci)
        bi = self._bmg.add_bernoulli(pi)
        si = self._bmg.add_sample(bi)
        return si

    def _replace_sample(self, node: bn.SampleNode) -> bn.BMGNode:
        dist = node.operand
        assert isinstance(dist, bn.BernoulliNode)
        size = self._typer[dist.probability]
        assert size == Size([2])
        s0 = self._add_sample(dist, 0)
        s1 = self._add_sample(dist, 1)
        t = self._bmg.add_tensor(size, s0, s1)
        self.fixed_one = True
        return t

    def _get_replacement(self, n: bn.BMGNode) -> Optional[bn.BMGNode]:
        if isinstance(n, bn.SampleNode):
            return self._replace_sample(n)
        return None


class VectorizedModelFixer:

    _bmg: BMGraphBuilder
    errors: ErrorReport

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        # We don't need the passed-in typer.
        self._bmg = bmg
        self.errors = ErrorReport()

    def fix_problems(self) -> None:
        vdf = VectorizedDistributionFixer(self._bmg, Sizer())
        vdf.fix_problems()
        assert not vdf.errors.any()

        if vdf.fixed_one:
            # We should now have one or more leaf sample nodes.
            for n in self._bmg.all_nodes():
                if vdf._needs_fixing(n):
                    assert n.is_leaf
                    self._bmg.remove_leaf(n)
