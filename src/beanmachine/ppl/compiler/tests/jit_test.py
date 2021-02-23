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


class C:
    def m(self):
        return


# TODO: support aliases on bm.random_variable

counter = 0

# Random variable that takes an argument
@bm.random_variable
def norm(n):
    global counter
    counter = counter + 1
    return Normal(loc=0.0, scale=1.0)


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


@bm.random_variable
def coin_with_class():
    C().m()
    f = True
    while f:
        f = not f
    return Beta(2.0, 2.0)


@bm.functional
def bad_functional_1():
    # It's not legal to call a random variable function with
    # a stochastic value that has infinite support.
    return norm(coin())


@bm.random_variable
def flips(n):
    return Bernoulli(0.5)


@bm.random_variable
def norm_ten(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
    return Normal(loc=0.0, scale=1.0)


@bm.functional
def bad_functional_2():
    # There are 1024 possibilities for this call; we give an
    # error when the control flow is this complex.
    return norm_ten(
        flips(0),
        flips(1),
        flips(2),
        flips(3),
        flips(4),
        flips(5),
        flips(6),
        flips(7),
        flips(8),
        flips(9),
    )


@bm.functional
def bad_functional_3():
    # Calling rv functions with named arguments is not allowed.
    return flips(n=1)


@bm.functional
def bad_functional_4():
    # Calling functionals with named arguments is not allowed.
    return exp_coin_2(c=1)


@bm.random_variable
def beta(n):
    return Beta(2.0, 2.0)


@bm.functional
def beta_tensor_1a():
    # What happens if we have two uses of the same RV indexed
    # with a tensor?
    return beta(tensor(1)).log()


@bm.functional
def beta_tensor_1b():
    return beta(tensor(1)).exp()


observable_side_effect = 0


def cause_side_effect():
    global observable_side_effect
    observable_side_effect = 1
    return True


@bm.random_variable
def assertions_are_removed():
    assert cause_side_effect()
    return Bernoulli(0.5)


@bm.random_variable
def assertions_are_removed():
    assert cause_side_effect()
    return Bernoulli(0.5)


@bm.random_variable
def spike_and_slab(n):
    if n:
        return Bernoulli(0.5)
    else:
        return Normal(0, 1)


# Try out a stochastic control flow where we choose
# a mean from one of two distributions depending on
# a coin flip.
@bm.random_variable
def choose_your_mean():
    return Normal(spike_and_slab(flip()), 1)


class JITTest(unittest.TestCase):
    def test_function_transformation_1(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        # Verify code generation of lifted, nested form of
        # functions f(x), norm(), above.

        self.assertTrue(norm.is_random_variable)

        bmgast, _ = _bm_function_to_bmg_ast(f, "f_helper")
        observed = astor.to_source(bmgast)
        expected = """
def f_helper(bmg):

    def f(x):
        a2 = bmg.handle_dot_get(math, 'exp')
        r3 = [x]
        r4 = {}
        r1 = bmg.handle_function(a2, r3, r4)
        return r1
    return f"""
        self.assertEqual(observed.strip(), expected.strip())

        bmgast, _ = _bm_function_to_bmg_ast(norm, "norm_helper")
        observed = astor.to_source(bmgast)
        expected = """
def norm_helper(bmg):

    def norm(n):
        global counter
        a1 = 1
        counter = bmg.handle_addition(counter, a1)
        r3 = []
        a6 = 0.0
        a5 = dict(loc=a6)
        a8 = 1.0
        a7 = dict(scale=a8)
        r4 = dict(**a5, **a7)
        r2 = bmg.handle_function(Normal, r3, r4)
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

    def test_function_transformation_6(self) -> None:
        """Unit tests for JIT functions"""

        # This test regresses some crashing bugs. The compiler crashed if an
        # RV function contained:
        #
        # * a class constructor
        # * a call to a class method
        # * a while loop

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [coin_with_class()]
        observations = {}
        bmg.accumulate_graph(queries, observations)
        dot = bmg.to_dot(point_at_input=True)
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1[label=alpha];
  N0 -> N1[label=beta];
  N1 -> N2[label=operand];
  N2 -> N3[label=operator];
}
"""
        self.assertEqual(dot.strip(), expected.strip())

    # TODO: Also test lambdas and nested functions.
    # TODO: What should we do about closures?

    def test_bad_control_flow_1(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [bad_functional_1()]
        observations = {}
        # TODO: Better exception class
        with self.assertRaises(ValueError) as ex:
            bmg.accumulate_graph(queries, observations)
        self.assertEqual(
            str(ex.exception), "Stochastic control flow must have finite support."
        )

    def test_bad_control_flow_2(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [bad_functional_2()]
        observations = {}
        # TODO: Better exception class
        with self.assertRaises(ValueError) as ex:
            bmg.accumulate_graph(queries, observations)
        self.assertEqual(str(ex.exception), "Stochastic control flow is too complex.")

    def test_bad_control_flow_3(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [bad_functional_3()]
        observations = {}
        # TODO: Better exception class
        with self.assertRaises(ValueError) as ex:
            bmg.accumulate_graph(queries, observations)
        self.assertEqual(
            str(ex.exception),
            "Random variable function calls must not have named arguments.",
        )

    def test_bad_control_flow_4(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [bad_functional_4()]
        observations = {}
        # TODO: Better exception class
        with self.assertRaises(ValueError) as ex:
            bmg.accumulate_graph(queries, observations)
        self.assertEqual(
            str(ex.exception),
            "Functional calls must not have named arguments.",
        )

    def test_rv_identity(self) -> None:
        self.maxDiff = None

        # This test demonstrates an invariant which we must maintain as we modify
        # the implementation details of the jitter: two calls to the same RV with
        # the same arguments must produce the same sample node.  Here the two calls
        # to beta(tensor(1)) must both produce the same sample node, not two samples.
        #
        # TODO:
        #
        # Right now this invariant is maintained by the @memoize modifier that is
        # automatically generated on a lifted rv function, but that mechanism
        # is redundant to the rv_map inside the graph builder, so we will eventually
        # remove it. When we do so, we'll need to ensure that one of the following
        # happens:
        #
        # * We add a hash function to RVIdentifier that treats identical-content tensors
        #   as the same argument
        # * We build a special-purpose map for tracking RVID -> Sample node mappings.
        # * We restrict arguments to rv functions to be hashable (and canonicalize tensor
        #   arguments to single values.)
        # * Or some other similar mechanism for maintaining this invariant.

        bmg = BMGraphBuilder()
        queries = [beta_tensor_1a(), beta_tensor_1b()]
        observations = {}
        bmg.accumulate_graph(queries, observations)
        observed = bmg.to_dot(point_at_input=True)
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Log];
  N4[label=Query];
  N5[label=Exp];
  N6[label=Query];
  N0 -> N1[label=alpha];
  N0 -> N1[label=beta];
  N1 -> N2[label=operand];
  N2 -> N3[label=operand];
  N2 -> N5[label=operand];
  N3 -> N4[label=operator];
  N5 -> N6[label=operator];
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_assertions_are_removed(self) -> None:
        # The lifted form of a function removes all assertion statements.
        # We can demonstrate this by making an assertion that causes an
        # observable effect, and then showing that the effect does not
        # happen when the lifted form is executed.
        global observable_side_effect
        self.maxDiff = None
        self.assertEqual(observable_side_effect, 0)

        # In non-lifted code, the assertion causes a side effect.
        assert cause_side_effect()
        self.assertEqual(observable_side_effect, 1)
        observable_side_effect = 0

        bmg = BMGraphBuilder()
        bmg.accumulate_graph([assertions_are_removed()], {})

        # The side effect is not caused.
        self.assertEqual(observable_side_effect, 0)

    def test_stochastic_control_flow_1(self) -> None:
        self.maxDiff = None

        bmg = BMGraphBuilder()
        queries = [choose_your_mean()]
        observations = {}
        bmg.accumulate_graph(queries, observations)
        # TODO: We cannot turn this into a BMG graph yet because we
        # do not support the map/index nodes. But we could turn this
        # particular model into an IF-THEN-ELSE control flow, which
        # we do support. Consider making that transformation when the
        # graph is accumulated.
        observed = bmg.to_dot(
            point_at_input=True, after_transform=False, label_edges=False
        )
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Bernoulli];
  N04[label=Sample];
  N05[label=0.0];
  N06[label=1.0];
  N07[label=Normal];
  N08[label=Sample];
  N09[label=0.5];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=map];
  N13[label=index];
  N14[label=1];
  N15[label=Normal];
  N16[label=Sample];
  N17[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N03;
  N03 -> N04;
  N04 -> N13;
  N05 -> N07;
  N05 -> N12;
  N06 -> N07;
  N06 -> N12;
  N07 -> N08;
  N08 -> N12;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
