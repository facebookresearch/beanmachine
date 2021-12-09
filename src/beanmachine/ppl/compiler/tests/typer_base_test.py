#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl.compiler.bmg_nodes as bn
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_node_types import is_supported_by_bmg
from beanmachine.ppl.compiler.typer_base import TyperBase


# To test out the typer base class, here is a very simple typer: it assigns
# the "type" True to a node if that node and *all* of its ancestors are supported
# node types for BMG, and False otherwise.
#
# The intention here is to demonstrate that the typer behaves as expected as we
# modify the graph and update the typer.


class SupportedTyper(TyperBase[bool]):
    def __init__(self) -> None:
        TyperBase.__init__(self)

    def _compute_type_inputs_known(self, node: bn.BMGNode) -> bool:
        return (isinstance(node, bn.ConstantNode) or is_supported_by_bmg(node)) and all(
            self[i] for i in node.inputs
        )


class TyperTest(unittest.TestCase):
    def test_typer(self) -> None:
        self.maxDiff = None
        # We start with this graph:
        #
        #  0  1
        #  |  |
        #  NORM
        #   |
        #   ~   2
        #   |   |
        #    DIV  3
        #     |   |
        #      ADD
        #     |   |
        #    EXP  NEG
        #
        # The DIV node is not supported in BMG.

        bmg = BMGraphBuilder()
        c0 = bmg.add_constant(0.0)
        c1 = bmg.add_constant(1.0)
        c2 = bmg.add_constant(2.0)
        c3 = bmg.add_constant(3.0)
        norm = bmg.add_normal(c0, c1)
        ns = bmg.add_sample(norm)
        d = bmg.add_division(ns, c2)
        a = bmg.add_addition(d, c3)
        e = bmg.add_exp(a)
        neg = bmg.add_negate(a)

        typer = SupportedTyper()

        # When we ask the typer for a judgment of a node, we should get judgments
        # of all of its ancestor nodes as well, but we skip computing types of
        # non-ancestors:

        self.assertTrue(typer[ns])  # Just type the sample and its ancestors.
        self.assertTrue(norm in typer)
        self.assertTrue(c0 in typer)
        self.assertTrue(c1 in typer)
        self.assertFalse(d in typer)
        self.assertFalse(a in typer)
        self.assertFalse(c2 in typer)
        self.assertFalse(c3 in typer)
        self.assertFalse(e in typer)
        self.assertFalse(neg in typer)

        # If we then type the exp, all of its ancestors become typed.
        # Division is not supported in BMG, so the division is marked
        # as not supported. The division is an ancestor of the addition
        # and exp, so they are typed as False also.

        self.assertFalse(typer[e])

        # The ancestors of the exp are now all typed.
        self.assertTrue(a in typer)
        self.assertTrue(d in typer)
        self.assertTrue(c3 in typer)
        self.assertTrue(c2 in typer)

        # But the negate is still not typed.
        self.assertFalse(neg in typer)

        # The types of the division, addition and exp are False:

        self.assertFalse(typer[d])
        self.assertFalse(typer[a])
        self.assertFalse(typer[e])
        self.assertTrue(typer[c2])
        self.assertTrue(typer[c3])

        # Now let's mutate the graph by adding some new nodes...
        c4 = bmg.add_constant(0.5)
        m = bmg.add_multiplication(ns, c4)
        # ... and mutating the addition:
        a.inputs[0] = m

        # The graph now looks like this:
        #
        #  0  1
        #  |  |
        #  NORM
        #   |
        #   ~    2
        #  | |   |
        #  |  DIV
        #  |
        #  |  0.5
        #  |   |
        #   MUL   3
        #     |   |
        #      ADD
        #     |   |
        #    EXP  NEG
        #
        # But we have not yet informed the typer that there was an update.

        self.assertFalse(typer[a])
        typer.update_type(a)
        self.assertTrue(typer[a])

        # This should trigger typing on the untyped ancestors of
        # the addition:

        self.assertTrue(m in typer)
        self.assertTrue(c4 in typer)

        # It should NOT trigger typing the NEG. We have yet to ask for the type of
        # that branch, so we do not spent time propagating type information down
        # the NEG branch.

        self.assertFalse(neg in typer)

        # The multiplication and exp should now all be marked as supported also.
        self.assertTrue(typer[m])
        self.assertTrue(typer[e])
