# Copyright (c) Facebook, Inc. and its affiliates.
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

        bmgast, _ = _bm_function_to_bmg_ast(y, "y_helper")
        observed = astor.to_source(bmgast)
        expected = """
def y_helper(bmg):

    def y():
        a1 = 0.0
        r4 = []
        r7 = {}
        a3 = bmg.handle_function(x, r4, r7)
        a5 = bmg.handle_less_than(a1, a3)
        if a5:
            a6 = 2.0
            z = bmg.handle_less_than(a3, a6)
        else:
            z = a5
        a13 = 3.0
        a10 = [a13]
        a14 = [z]
        a9 = bmg.handle_addition(a10, a14)
        a15 = 4.0
        a11 = [a15]
        r8 = bmg.handle_addition(a9, a11)
        r12 = {}
        r2 = bmg.handle_function(StudentT, r8, r12)
        return r2
    return y
"""
        self.assertEqual(observed.strip(), expected.strip())
