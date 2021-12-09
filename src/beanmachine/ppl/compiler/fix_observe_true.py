# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BMGNode,
    ExpNode,
    Observation,
    SampleNode,
    ToIntNode,
    ToPositiveRealNode,
    ToProbabilityNode,
    ToRealNode,
    UnaryOperatorNode,
)
from beanmachine.ppl.compiler.bmg_types import is_one
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.typer_base import TyperBase


def _is_conversion(n: BMGNode) -> bool:
    return (
        isinstance(n, ToPositiveRealNode)
        or isinstance(n, ToProbabilityNode)
        or isinstance(n, ToRealNode)
        or isinstance(n, ToIntNode)
    )


def _skip_conversions(n: BMGNode) -> BMGNode:
    while _is_conversion(n):
        assert isinstance(n, UnaryOperatorNode)
        n = n.operand
    return n


class ObserveTrueFixer:
    # A common technique in model design is to boost the probability density
    # score of a particular quantity by converting it to a probability
    # and then observing that a coin flip of that probability comes up heads.
    # This should be logically equivalent to boosting by adding an EXP_PRODUCT
    # factor, but when we run models like that through BMG inference, we're
    # getting different results than when we add a factor.
    #
    # To work around the problem while we diagnose it we can use this fixer.
    # It looks for graphs of the form:
    #
    #      SOMETHING --> EXP --> TO_PROB --> BERNOULLI --> SAMPLE --> OBSERVE TRUE
    #
    # and converts them to
    #
    #      SOMETHING --> EXP --> TO_PROB --> BERNOULLI --> SAMPLE
    #        \
    #         --> EXP_PRODUCT

    bmg: BMGraphBuilder
    errors: ErrorReport
    _typer: TyperBase

    def __init__(self, bmg: BMGraphBuilder, typer: TyperBase) -> None:
        # The typer is not actually needed but the caller assumes that
        # all problem fixers need a typer to either get types or propagate
        # updates. This fixer does neither, since it only works on leaf nodes
        # and types of values.
        self.bmg = bmg
        self.errors = ErrorReport()
        self._typer = typer

    def _fix_observation(self, o: Observation) -> None:
        if not is_one(o.value):
            return
        sample = o.observed
        if not isinstance(sample, SampleNode):
            return
        bern = sample.operand
        if not isinstance(bern, BernoulliNode):
            return
        exp = _skip_conversions(bern.probability)
        if not isinstance(exp, ExpNode):
            return
        self.bmg.add_exp_product(exp.operand)
        self.bmg.remove_leaf(o)

    def fix_problems(self) -> None:
        for o in self.bmg.all_observations():
            self._fix_observation(o)
