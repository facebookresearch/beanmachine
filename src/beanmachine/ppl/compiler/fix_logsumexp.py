# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)


def logsumexp_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    """This fixer attempts to rewrite log expressions of the form
    log( exp(a) + exp(b) + exp(c) ...) -> logsumexp(a,b,c, ...)
    """

    def logsumexp_fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.LogNode):
            return Inapplicable
        addition = node.operand
        if not isinstance(addition, bn.AdditionNode):
            return Inapplicable
        if not all(isinstance(i, bn.ExpNode) for i in addition.inputs):
            return Inapplicable
        return bmg.add_logsumexp(*[i.operand for i in addition.inputs])

    return logsumexp_fixer
