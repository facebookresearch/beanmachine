# Copyright (c) Facebook, Inc. and its affiliates.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BMGNode,
    ExpNode,
    Observation,
    SampleNode,
    ToPositiveRealNode,
    ToProbabilityNode,
    ToRealNode,
    UnaryOperatorNode,
)
from beanmachine.ppl.compiler.bmg_types import Boolean
from beanmachine.ppl.compiler.error_report import ErrorReport


def _is_conversion(n: BMGNode) -> bool:
    return (
        isinstance(n, ToPositiveRealNode)
        or isinstance(n, ToProbabilityNode)
        or isinstance(n, ToRealNode)
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
    errors = ErrorReport()

    def __init__(self, bmg: BMGraphBuilder) -> None:
        self.bmg = bmg

    def _fix_observation(self, o: Observation) -> None:
        if o.graph_type != Boolean or not o.value:
            return
        sample = o.operand
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
