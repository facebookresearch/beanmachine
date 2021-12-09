# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch import tensor
from torch.distributions import Bernoulli, Beta, Normal


def f(x):
    return math.exp(x)


class C:
    def m(self):
        return


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
def flip_with_comprehension():
    _ = [0 for x in []]
    return Bernoulli(0.5)


@bm.random_variable
def flip_with_nested_function():
    def x():
        return 0.5

    x()
    return Bernoulli(0.5)


# Verify that aliased decorator is allowed:

myrv = bm.random_variable


@myrv
def aliased_rv():
    return Bernoulli(0.5)


# Verify that random variable constructed without explicit decorator is allowed:


def some_function():
    return Bernoulli(0.5)


undecorated_rv = myrv(some_function)


# TODO: What if some_function is a lambda instead of a function definition?
# TODO: What if the function has outer variables?


class JITTest(unittest.TestCase):
    def test_function_transformation_1(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        # Verify code generation of lifted, nested form of
        # functions f(x), norm(), above.

        self.assertTrue(norm.is_random_variable)

        # TODO: Stop using _bm_function_to_bmg_ast for testing.
        bmgast = _bm_function_to_bmg_ast(f, "f_helper")
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

        bmgast = _bm_function_to_bmg_ast(norm().function, "norm_helper")
        observed = astor.to_source(bmgast)
        expected = """
def norm_helper(bmg):

    def norm(n):
        global counter
        a1 = 1
        counter = bmg.handle_addition(counter, a1)
        r4 = []
        a9 = 0.0
        a8 = dict(loc=a9)
        a11 = 1.0
        a10 = dict(scale=a11)
        r7 = dict(**a8, **a10)
        r2 = bmg.handle_function(Normal, r4, r7)
        return r2
    a3 = bmg.handle_dot_get(bm, 'random_variable')
    r5 = [norm]
    r6 = {}
    norm = bmg.handle_function(a3, r5, r6)
    return norm
"""
        self.assertEqual(observed.strip(), expected.strip())

        # * Obtain the lifted version of f.
        # * Ask the graph builder to transform the rv associated
        #   with norm(0) to a sample node.
        # * Invoke the lifted f and verify that we accumulate an
        #   exp(sample(normal(0, 1))) node into the graph.

        bmg = BMGRuntime()

        lifted_f = _bm_function_to_bmg_function(f, bmg)
        norm_sample = bmg._rv_to_node(norm(0))

        result = lifted_f(norm_sample)
        self.assertTrue(isinstance(result, ExpNode))
        dot = to_dot(bmg._bmg)
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

        bmg = BMGRuntime()
        bmg._rv_to_node(flip())
        dot = to_dot(bmg._bmg)
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

        rt = BMGRuntime()
        queries = [coin(), exp_coin()]
        observations = {flip(): tensor(1.0)}
        bmg = rt.accumulate_graph(queries, observations)
        dot = to_dot(bmg)
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

        rt = BMGRuntime()
        queries = [exp_norm(0)]
        observations = {}
        bmg = rt.accumulate_graph(queries, observations)
        dot = to_dot(bmg)
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

        rt = BMGRuntime()
        queries = [exp_coin_3()]
        observations = {}
        bmg = rt.accumulate_graph(queries, observations)
        dot = to_dot(bmg)

        # Note that though functional exp_coin_3 calls functional exp_coin_2,
        # we only get one query node emitted into the graph because the
        # caller only asked for one query node.

        expected = """
digraph "graph" {
  N0[label=1];
  N1[label=2.0];
  N2[label=Beta];
  N3[label=Sample];
  N4[label=Exp];
  N5[label="+"];
  N6[label=Query];
  N0 -> N5[label=left];
  N1 -> N2[label=alpha];
  N1 -> N2[label=beta];
  N2 -> N3[label=operand];
  N3 -> N4[label=operand];
  N4 -> N5[label=right];
  N5 -> N6[label=operator];
}
"""
        self.assertEqual(expected.strip(), dot.strip())

    def test_function_transformation_6(self) -> None:
        """Unit tests for JIT functions"""

        # This test regresses some crashing bugs. The compiler crashed if an
        # RV function contained:
        #
        # * a class constructor
        # * a call to a class method
        # * a while loop

        self.maxDiff = None

        rt = BMGRuntime()
        queries = [coin_with_class()]
        observations = {}
        bmg = rt.accumulate_graph(queries, observations)
        dot = to_dot(bmg)
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

        bmg = BMGRuntime()
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

        bmg = BMGRuntime()
        queries = [bad_functional_2()]
        observations = {}
        # TODO: Better exception class
        with self.assertRaises(ValueError) as ex:
            bmg.accumulate_graph(queries, observations)
        self.assertEqual(str(ex.exception), "Stochastic control flow is too complex.")

    def test_bad_control_flow_3(self) -> None:
        """Unit tests for JIT functions"""

        self.maxDiff = None

        bmg = BMGRuntime()
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

        bmg = BMGRuntime()
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

        rt = BMGRuntime()
        queries = [beta_tensor_1a(), beta_tensor_1b()]
        observations = {}
        bmg = rt.accumulate_graph(queries, observations)
        observed = to_dot(bmg)
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

        bmg = BMGRuntime()
        bmg.accumulate_graph([assertions_are_removed()], {})

        # The side effect is not caused.
        self.assertEqual(observable_side_effect, 0)

    def test_nested_functions_and_comprehensions(self) -> None:
        self.maxDiff = None

        # We had a bug where a nested function or comprehension inside a
        # random_variable would crash while accumulating the graph;
        # this test regresses that bug by simply verifying that we do
        # not crash in those scenarios now.

        bmg = BMGRuntime()
        bmg.accumulate_graph([flip_with_nested_function()], {})

        bmg = BMGRuntime()
        bmg.accumulate_graph([flip_with_comprehension()], {})

    def test_aliased_rv(self) -> None:
        self.maxDiff = None
        rt = BMGRuntime()
        queries = [aliased_rv()]
        observations = {}
        bmg = rt.accumulate_graph(queries, observations)
        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1[label=probability];
  N1 -> N2[label=operand];
  N2 -> N3[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_undecorated_rv(self) -> None:
        self.maxDiff = None
        rt = BMGRuntime()
        queries = [undecorated_rv()]
        observations = {}
        bmg = rt.accumulate_graph(queries, observations)
        observed = to_dot(bmg)
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1[label=probability];
  N1 -> N2[label=operand];
  N2 -> N3[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_nested_rv(self) -> None:
        self.maxDiff = None

        # A random variable that is nested inside another function and closed over
        # an outer local variable works:

        prob = 0.5

        @bm.random_variable
        def nested_flip():
            return Bernoulli(prob)

        observed = to_dot(BMGRuntime().accumulate_graph([nested_flip()], {}))
        expected = """
digraph "graph" {
  N0[label=0.5];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1[label=probability];
  N1 -> N2[label=operand];
  N2 -> N3[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # Check that we can close over an outer variable with a lambda also.
        mean = 0.0
        sigma = 1.0
        shift = 2.0
        # Lambda random variable:
        lambda_norm = bm.random_variable(lambda: Normal(mean, sigma))
        # Lambda that is not a functional, not declared inside a functional, but called
        # from inside a functional:
        lambda_mult = lambda x, y: x * y  # noqa
        # Lambda functional:
        lambda_sum = bm.functional(lambda: lambda_mult(lambda_norm() + shift, 4.0))
        observed = to_dot(BMGRuntime().accumulate_graph([lambda_sum()], {}))
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=2.0];
  N5[label="+"];
  N6[label=4.0];
  N7[label="*"];
  N8[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N3 -> N5[label=left];
  N4 -> N5[label=right];
  N5 -> N7[label=left];
  N6 -> N7[label=right];
  N7 -> N8[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # What if we have a nested function inside a random variable?

        @bm.random_variable
        def norm1():
            return Normal(0.0, 1.0)

        @bm.random_variable
        def norm2():
            def mult(x, y):
                return x * y

            return Normal(mult(norm1(), 2.0), 3.0)

        observed = to_dot(BMGRuntime().accumulate_graph([norm2()], {}))
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=2.0];
  N5[label="*"];
  N6[label=3.0];
  N7[label=Normal];
  N8[label=Sample];
  N9[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N3 -> N5[label=left];
  N4 -> N5[label=right];
  N5 -> N7[label=mu];
  N6 -> N7[label=sigma];
  N7 -> N8[label=operand];
  N8 -> N9[label=operator];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # What if we have a random variable nested inside another?

        @bm.random_variable
        def norm3():
            @bm.random_variable
            def norm4():
                return Normal(0.0, 1.0)

            return Normal(norm4() * 5.0, 6.0)

        observed = to_dot(BMGRuntime().accumulate_graph([norm3()], {}))
        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=5.0];
  N5[label="*"];
  N6[label=6.0];
  N7[label=Normal];
  N8[label=Sample];
  N9[label=Query];
  N0 -> N2[label=mu];
  N1 -> N2[label=sigma];
  N2 -> N3[label=operand];
  N3 -> N5[label=left];
  N4 -> N5[label=right];
  N5 -> N7[label=mu];
  N6 -> N7[label=sigma];
  N7 -> N8[label=operand];
  N8 -> N9[label=operator];
}

"""
        self.assertEqual(expected.strip(), observed.strip())
