# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import math
import unittest

import astor
import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_to_bmg import (
    _bm_function_to_bmg_ast,
    _bm_function_to_bmg_function,
)
from beanmachine.ppl.compiler.bmg_nodes import ExpNode
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from torch.distributions import Normal


def f(x):
    return math.exp(x)


# TODO: support aliases on bm.random_variable

counter = 0


@bm.random_variable
def norm(n):
    global counter
    counter = counter + 1
    return Normal(0.0, 1.0)


class JITTest(unittest.TestCase):
    def test_function_transformation_1(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        # Verify code generation of lifted, nested form of
        # functions f(x), norm(), above.

        self.assertTrue(norm.is_random_variable)

        bmgast = _bm_function_to_bmg_ast(f, "f_helper")
        observed = astor.to_source(bmgast)
        expected = """
def f_helper(bmg):
    from beanmachine.ppl.utils.memoize import memoize
    from beanmachine.ppl.utils.probabilistic import probabilistic

    def f(x):
        a2 = bmg.handle_dot_get(math, 'exp')
        r3 = [x]
        r4 = {}
        r1 = bmg.handle_function(a2, [*r3], r4)
        return r1
    return f"""
        self.assertEqual(observed.strip(), expected.strip())

        bmgast = _bm_function_to_bmg_ast(norm, "norm_helper")
        observed = astor.to_source(bmgast)
        expected = """
def norm_helper(bmg):
    from beanmachine.ppl.utils.memoize import memoize
    from beanmachine.ppl.utils.probabilistic import probabilistic

    @probabilistic(bmg)
    @memoize
    def norm(n):
        global counter
        a1 = 1
        counter = bmg.handle_addition(counter, a1)
        a5 = 0.0
        a4 = [a5]
        a8 = 1.0
        a6 = [a8]
        r3 = bmg.handle_addition(a4, a6)
        r7 = {}
        r2 = bmg.handle_function(Normal, [*r3], r7)
        return bmg.handle_sample(r2)
    return norm
"""
        self.assertEqual(observed.strip(), expected.strip())

        # * Obtain the lifted version of f.
        # * Ask the graph builder to transform the rv associated
        #   with norm(0) to a sample node.
        # * Invoke the lifted f and verify that we accumulate an
        #   exp(sample(normal(0, 1))) node into the graph.

        bmg = BMGraphBuilder()

        lifted_f = _bm_function_to_bmg_function(f, bmg)
        norm_sample = bmg._rv_to_node(norm(0))

        result = lifted_f(norm_sample)
        self.assertTrue(isinstance(result, ExpNode))
        dot = bmg.to_dot(point_at_input=True)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Exp];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N3 -> N4[label=operand];
}
"""
        self.assertEqual(dot.strip(), expected.strip())

        # Verify that we've executed the body of the lifted
        # norm(n) exactly once.
        global counter
        self.assertEqual(counter, 1)

        # Turning an rv into a node should be idempotent;
        # the second time, we do not increment the counter.

        bmg._rv_to_node(norm(0))
        self.assertEqual(counter, 1)
        bmg._rv_to_node(norm(1))
        self.assertEqual(counter, 2)
        bmg._rv_to_node(norm(1))
        self.assertEqual(counter, 2)
