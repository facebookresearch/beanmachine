# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import beanmachine.ppl.compiler.bmg_nodes as bn
import beanmachine.ppl.compiler.bmg_types as bt
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problem import (
    Inapplicable,
    NodeFixer,
    NodeFixerResult,
)
from beanmachine.ppl.compiler.lattice_typer import LatticeTyper


# TODO: Move this to a utilities module
def _count(bs) -> int:
    """Given a sequence of bools, count the Trues"""
    return sum(1 for b in bs if b)


def negative_real_multiplication_fixer(
    bmg: BMGraphBuilder, typer: LatticeTyper
) -> NodeFixer:
    """This fixer rewrites multiplications involving negative reals into
    multiplications using only positive reals."""

    # The BMG type system requires that all inputs to a multiplication be of
    # the same type and that the type is a floating point type. The output type
    # is then the same as the input type. There are three possibilities:
    #
    # P  * P  --> P
    # R  * R  --> R
    # R+ * R+ --> R+
    #
    # This means that if we have MULT(R-, R+) then the requirements fixer will
    # convert both inputs to R, and we will lose track of the fact that
    # we could know in the type system that the result is a R-.
    #
    # This is particularly unfortunate when the negative real is a log probability.
    # If we multiply a log probability by two, say, that is logically squaring
    # the probability. We know that POW(P, 2) --> P, and MULT(P, P) --> P, but
    # the BMG multiplication operator does not know that MULT(2, R-) is R-.
    #
    # We could solve this problem by modifying the BMG type system so that we
    # allow mixed-type inputs to a multiplication; until we do so, we'll
    # work around the problem here by rewriting multiplications that involve
    # combinations of R-, R+ and P.

    def _negative_real_multiplication_fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.MultiplicationNode):
            return Inapplicable

        # If no input is R- then we don't have a rewrite to do here.
        count = _count(typer[inpt] == bt.NegativeReal for inpt in node.inputs)
        if count == 0:
            return Inapplicable

        # If any input is R, we cannot prevent the output from being R.
        if any(typer[inpt] == bt.Real for inpt in node.inputs):
            return Inapplicable

        new_mult = bmg.add_multi_multiplication(
            *(
                bmg.add_negate(inpt) if typer[inpt] == bt.NegativeReal else inpt
                for inpt in node.inputs
            )
        )

        if count % 2 != 0:
            new_mult = bmg.add_negate(new_mult)
        return new_mult

    return _negative_real_multiplication_fixer


def _input_of_1m_exp(node: bn.BMGNode) -> Optional[bn.BMGNode]:
    # Is node ADD(1, NEG(EXP(X))) or ADD(NEG(EXP(X)), 1)?
    if not isinstance(node, bn.AdditionNode) or len(node.inputs) != 2:
        return None
    left = node.inputs[0]
    right = node.inputs[1]
    negate = None
    if bn.is_one(left):
        negate = right
    elif bn.is_one(right):
        negate = left
    if not isinstance(negate, bn.NegateNode):
        return None
    ex = negate.inputs[0]
    if not isinstance(ex, bn.ExpNode):
        return None
    return ex.inputs[0]


def log1mexp_fixer(bmg: BMGraphBuilder, typer: LatticeTyper) -> NodeFixer:
    # To take the complement of a log prob we need to convert
    # the log prob back to an ordinary prob, then complement it,
    # then convert it back to a log prob. In BMG we have a special
    # node just for this operation.

    def _is_comp_exp(node: bn.BMGNode) -> bool:
        return isinstance(node, bn.ComplementNode) and isinstance(
            node.inputs[0], bn.ExpNode
        )

    def _log1mexp_fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.LogNode):
            return Inapplicable

        comp = node.inputs[0]
        # There are two situations to consider here.
        #
        # Easy case:
        #
        # If we've already rewritten the 1-exp(x) into
        # complement(x), then we already know that x is
        # a probability. Just generate log1mexp(x).
        #
        # Hard case:
        #
        # We sometimes get into a situation where you and I know that
        # a node is a probability, but the type checker does not.
        # For example, if we have probability p1:
        #
        # p2 = 0.5 + p1 / 2
        #
        # then p2 is judged to be R+ because the sum of two Ps is not necessarily a P.
        #
        # If we then go on to do:
        #
        # x = log(p2)
        #
        # then x is NOT known to be R-; rather it is known to be R.
        #
        # If later on we wish to invert this log prob:
        #
        # inv = log(1 - exp(x))
        #
        # Then the type system says exp(x) is R+ (when it should be P).
        # We then say that 1 - R+ is R (should be P) and then log(R) is an
        # error, when it should be R-.
        #
        # If the program has log(1-exp(x) then the developer certainly
        # believes that x is a negative real. Even if the type system
        # does not, we should generate a graph as though this were a
        # negative real.
        x = None
        if _is_comp_exp(comp):
            x = comp.inputs[0].inputs[0]
        else:
            x = _input_of_1m_exp(comp)
            # If x is known to be a positive real, there's nothing
            # we can do. A later pass will give an error.
            #
            # If it is real, then force it to be negative real.
            if x is not None:
                if typer.is_pos_real(x):
                    return Inapplicable
                if typer.is_real(x):
                    x = bmg.add_to_negative_real(x)
        # If x is None then the node does not match log(1-exp(x)).
        # If x is not None, it still might be untypable. Skip doing
        # this rewrite until we know that x has a type.
        if x is None or not typer.is_neg_real(x):
            return Inapplicable
        return bmg.add_log1mexp(x)

    return _log1mexp_fixer


def neg_neg_fixer(bmg: BMGraphBuilder) -> NodeFixer:
    # We can easily end up in a situation where another rewriter causes
    # the graph to contain X -->  NEG  -->  NEG  which could be replaced
    # with just X.

    def _neg_neg_fixer(node: bn.BMGNode) -> NodeFixerResult:
        if not isinstance(node, bn.NegateNode):
            return Inapplicable
        neg = node.inputs[0]
        if not isinstance(neg, bn.NegateNode):
            return Inapplicable
        return neg.inputs[0]

    return _neg_neg_fixer
