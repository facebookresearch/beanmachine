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
# The tensor probability can be of any length > 1 but it must be one-dimensional.
#
# TODO: Implement the same logic for two-dimensional probabilities.


def _is_fixable_size(s: Size) -> bool:
    return len(s) == 1 and s[0] > 1


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
        return _is_fixable_size(s)

    def _add_sample(self, node: bn.BernoulliNode, i: int) -> bn.SampleNode:
        p = node.probability
        size = self._typer[p]
        assert _is_fixable_size(size)
        ci = self._bmg.add_constant(i)
        pi = self._bmg.add_index(p, ci)
        bi = self._bmg.add_bernoulli(pi)
        si = self._bmg.add_sample(bi)
        return si

    def _replace_sample(self, node: bn.SampleNode) -> bn.BMGNode:
        dist = node.operand
        assert isinstance(dist, bn.BernoulliNode)
        size = self._typer[dist.probability]
        assert _is_fixable_size(size)
        samples = []
        for i in range(0, size[0]):
            samples.append(self._add_sample(dist, i))
        t = self._bmg.add_tensor(size, *samples)
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

    def _fix_observations(self) -> None:
        for o in self._bmg.all_observations():
            observed = o.observed
            if not isinstance(observed, bn.TensorNode):
                continue
            if not _is_fixable_size(observed._size):
                continue
            # TODO: What if the observation is of a different size than the
            # tensor node we've just generated? That should be an error, but instead
            # we just crash here. Figure out where to put an error detection pass
            # which prevents this crash and reports the error.
            for i in range(0, len(observed.inputs)):
                s = observed.inputs[i]
                assert isinstance(s, bn.SampleNode)
                self._bmg.add_observation(s, o.value[i])
            self._bmg.remove_leaf(o)

    def fix_problems(self) -> None:
        vdf = VectorizedDistributionFixer(self._bmg, Sizer())
        vdf.fix_problems()
        assert not vdf.errors.any()

        if not vdf.fixed_one:
            # We changed nothing so there is nothing more to do.
            return

        # We changed something. We should now have one or more leaf
        # sample nodes.
        for n in self._bmg.all_nodes():
            if vdf._needs_fixing(n):
                assert n.is_leaf
                self._bmg.remove_leaf(n)

        # We might have an illegal observation. Fix it.
        self._fix_observations()
