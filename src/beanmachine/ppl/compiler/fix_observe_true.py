# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BMGNode,
    ExpNode,
    SampleNode,
    ToIntNode,
    ToPositiveRealNode,
    ToProbabilityNode,
    ToRealNode,
    UnaryOperatorNode,
)
from beanmachine.ppl.compiler.bmg_types import is_one
from beanmachine.ppl.compiler.error_report import ErrorReport
from beanmachine.ppl.compiler.fix_problem import GraphFixer, GraphFixerResult


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


def observe_true_fixer(bmg: BMGraphBuilder) -> GraphFixer:
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
    def fixer() -> GraphFixerResult:
        made_change = False
        for o in bmg.all_observations():
            if not is_one(o.value):
                continue
            sample = o.observed
            if not isinstance(sample, SampleNode):
                continue
            bern = sample.operand
            if not isinstance(bern, BernoulliNode):
                continue
            exp = _skip_conversions(bern.probability)
            if not isinstance(exp, ExpNode):
                continue
            bmg.add_exp_product(exp.operand)
            bmg.remove_leaf(o)
            made_change = True
        return made_change, ErrorReport()

    return fixer
