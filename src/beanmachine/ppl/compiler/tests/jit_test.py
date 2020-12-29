# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import math
import unittest

import astor
import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bm_to_bmg import (
    _bm_function_to_bmg_ast,
    _bm_function_to_bmg_function,
)
from beanmachine.ppl.compiler.bmg_nodes import ExpNode
from torch import tensor
from torch.distributions import Bernoulli, Beta, Normal


def f(x):
    return math.exp(x)


# TODO: support aliases on bm.random_variable

counter = 0

# Random variable that takes an argument
@bm.random_variable
def norm(n):
    global counter
    counter = counter + 1
    return Normal(0.0, 1.0)


# Random variable that takes no argument
@bm.random_variable
def coin():
    return Beta(2.0, 2.0)


# Call to random variable inside random variable
@bm.random_variable
def flip():
    return Bernoulli(coin())


# Functional that takes no argument
@bm.functional
def exp_coin():
    return coin().exp()


# Functional that takes an ordinary value argument
@bm.functional
def exp_norm(n):
    return norm(n).exp()


# Functional that takes an graph node argument
@bm.functional
def exp_coin_2(c):
    return c.exp()


# Ordinary function
def add_one(x):
    return 1 + x


# Functional that calls normal, functional, random variable functions
@bm.functional
def exp_coin_3():
    return add_one(exp_coin_2(coin()))


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

    def test_function_transformation_2(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        # We have flip() which calls Bernoulli(coin()). What should happen
        # here is:
        # * _rv_to_node jit-compiles flip() and executes the lifted version.
        # * while executing the lifted flip() we encounter a call to
        #   coin().  We detect that coin is a random variable function,
        #   and call it.
        # * We now have the RVIdentifier for coin() in hand, which we
        #   then jit-compile in turn, and execute the lifted version.
        # * That completes the construction of the graph.

        bmg = BMGraphBuilder()
        bmg._rv_to_node(flip())
        dot = bmg.to_dot(point_at_input=True)
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Bernoulli];
  N4[label=Sample];
  N0 -> N1[label=alpha];
  N0 -> N1[label=beta];
  N1 -> N2[label=operand];
  N2 -> N3[label=probability];
  N3 -> N4[label=operand];
}
"""
        self.assertEqual(dot.strip(), expected.strip())

    def test_function_transformation_3(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [coin(), exp_coin()]
        observations = {flip(): tensor(1.0)}
        bmg.accumulate_graph(queries, observations)
        dot = bmg.to_dot(point_at_input=True)
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Bernoulli];
  N4[label=Sample];
  N5[label="Observation tensor(1.)"];
  N6[label=Query];
  N7[label=Exp];
  N8[label=Query];
  N0 -> N1[label=alpha];
  N0 -> N1[label=beta];
  N1 -> N2[label=operand];
  N2 -> N3[label=probability];
  N2 -> N6[label=operator];
  N2 -> N7[label=operand];
  N3 -> N4[label=operand];
  N4 -> N5[label=operand];
  N7 -> N8[label=operator];
}
"""
        self.assertEqual(dot.strip(), expected.strip())

    def test_function_transformation_4(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [exp_norm(0)]
        observations = {}
        bmg.accumulate_graph(queries, observations)
        dot = bmg.to_dot(point_at_input=True)
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=Exp];
  N5[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N3 -> N4[label=operand];
  N4 -> N5[label=operator];
}
"""
        self.assertEqual(dot.strip(), expected.strip())

    def test_function_transformation_5(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [exp_coin_3()]
        observations = {}
        bmg.accumulate_graph(queries, observations)
        dot = bmg.to_dot(point_at_input=True)

        # Note that though functional exp_coin_3 calls functional exp_coin_2,
        # we only get one query node emitted into the graph because the
        # caller only asked for one query node.

        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Exp];
  N4[label=1];
  N5[label="+"];
  N6[label=Query];
  N0 -> N1[label=alpha];
  N0 -> N1[label=beta];
  N1 -> N2[label=operand];
  N2 -> N3[label=operand];
  N3 -> N5[label=right];
  N4 -> N5[label=left];
  N5 -> N6[label=operator];
}
"""
        self.assertEqual(dot.strip(), expected.strip())


# TODO: Also test lambdas and nested functions.
# TODO: What should we do about closures?
