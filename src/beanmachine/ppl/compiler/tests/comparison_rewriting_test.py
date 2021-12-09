# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Comparison operators are not supported in BMG yet but we need to be able to detect
use of them and give an error. Here we verify that we can rewrite code containing
them correctly."""

import unittest

import astor
import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_to_bmg import _bm_function_to_bmg_ast
from torch.distributions import Normal, StudentT


@bm.random_variable
def x():
    return Normal(0.0, 1.0)


@bm.random_variable
def y():
    z = 0.0 < x() < 2.0
    return StudentT(3.0, z, 4.0)


class ComparisonRewritingTest(unittest.TestCase):
    def test_comparison_rewriting_1(self) -> None:
        self.maxDiff = None

        # The key thing to note here is that we eliminate Python's weird
        # comparison logic entirely; we reduce
        #
        # z = 0.0 < x() < 2.0
        #
        # to the equivalent of:
        #
        # tx = x()
        # comp = 0.0 < tx
        # if comp:
        #   z = tx < 2.0
        # else:
        #   z = comp
        #
        # which has the same semantics but has only simple comparisons and
        # simple control flows.

        self.assertTrue(y.is_random_variable)

        bmgast = _bm_function_to_bmg_ast(y().function, "y_helper")
        observed = astor.to_source(bmgast)
        expected = """
def y_helper(bmg):

    def y():
        a1 = 0.0
        r6 = []
        r10 = {}
        a4 = bmg.handle_function(x, r6, r10)
        a7 = bmg.handle_less_than(a1, a4)
        bmg.handle_if(a7)
        if a7:
            a8 = 2.0
            z = bmg.handle_less_than(a4, a8)
        else:
            z = a7
        a16 = 3.0
        a13 = [a16]
        a17 = [z]
        a12 = bmg.handle_addition(a13, a17)
        a18 = 4.0
        a14 = [a18]
        r11 = bmg.handle_addition(a12, a14)
        r15 = {}
        r2 = bmg.handle_function(StudentT, r11, r15)
        return r2
    a3 = bmg.handle_dot_get(bm, 'random_variable')
    r5 = [y]
    r9 = {}
    y = bmg.handle_function(a3, r5, r9)
    return y
"""
        self.assertEqual(observed.strip(), expected.strip())
