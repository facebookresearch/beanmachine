# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
