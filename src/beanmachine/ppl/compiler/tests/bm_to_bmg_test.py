# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for bm_to_bmg.py"""
import unittest

import astor
import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_to_bmg import _bm_function_to_bmg_ast
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Normal, Dirichlet, Bernoulli


class BaseModel:
    @bm.random_variable
    def normal(self):
        return Normal(0.0, 1.0)

    @bm.functional
    def foo(self):
        return self.normal() + 2.0

    def bar(self):
        return 3.0


class DerivedModel(BaseModel):
    @bm.functional
    def foo(self):
        f = super().foo()
        b = super(DerivedModel, self).bar()
        return f * b  # This should be (n() + 2) * 3

    def bar(self):
        return 4.0


@bm.random_variable
def legal_subscript_mutations():
    t = tensor([0.0, 0.0])
    t[0] = 0.0
    t[1] = 1.0
    t[0:] = 2.0
    t[:1] = 3.0
    t[0:1] = 4.0
    t[0::] = 5.0
    t[:1:] = 6.0
    t[::1] = 7.0
    t[0:1:] = 8.0
    t[0::1] = 9.0
    t[:1:1] = 10.0
    t[0:1:1] = 11.0
    return Dirichlet(t)


@bm.random_variable
def normal():
    return Normal(0.0, 1.0)


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def illegal_subscript_mutation_1():
    # Mutate a tensor with a stochastic value:
    t = tensor([0.0, 0.0])
    t[0] = normal()
    return t


@bm.functional
def illegal_subscript_mutation_2():
    # Mutate a stochastic tensor
    t = legal_subscript_mutations()
    t[0] = 0.0
    return t


@bm.functional
def illegal_subscript_mutation_3():
    # Mutate a tensor with a stochastic index
    t = tensor([0.0, 0.0])
    t[flip()] = 1.0
    return t


@bm.functional
def illegal_subscript_mutation_4():
    # Mutate a tensor with a stochastic upper
    t = tensor([0.0, 0.0])
    t[0 : flip()] = 1.0
    return t


@bm.functional
def illegal_subscript_mutation_5():
    # Mutate a tensor with a stochastic step
    t = tensor([0.0, 0.0])
    t[0 : 1 : flip() + 1] = 1.0
    return t


class CompilerTest(unittest.TestCase):
    def test_super_call(self) -> None:
        self.maxDiff = None
        # A call to super() in Python is not a normal function. Consider:
        def outer(s):
            return s().x()

        class B:
            def x(self):
                return 1

        class D(B):
            def x(self):
                return 2

            def ordinary(self):
                return self.x()  # 2

            def sup1(self):
                return super().x()  # 1

            def sup2(self):
                s = super
                return s().x()  # Doesn't have to be a keyword

            def callout(self):
                return outer(super)  # but the call to super() needs to be inside D.

        self.assertEqual(D().ordinary(), 2)
        self.assertEqual(D().sup1(), 1)
        self.assertEqual(D().sup2(), 1)
        # What's happening here is: "super()" is a syntactic sugar for "super(__class__, self)"
        # where __class__ is an automatically-generated outer variable of the method that
        # contains the call to super(). That variable has the value of the containing class.
        # When we call D().callout() here, there is no automatically-generated outer variable
        # when super() is ultimately called, and therefore we get this confusing but expected
        # exception raised:
        with self.assertRaises(RuntimeError) as ex:
            D().callout()
        expected = "super(): __class__ cell not found"
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())

        # bm_to_bmg rewrites all random variables, all functionals, and their callees.
        # We must ensure that all calls to super() are (1) syntactically exactly that;
        # these calls must not be rewritten to bmg.handle_call, and (2) must have an
        # outer variable __class__ which is initialized to the class which originally
        # declared the random variable.

        d = DerivedModel()
        observed = BMGInference().to_dot([d.foo()], {})

        expected = """
digraph "graph" {
  N0[label=0.0];
  N1[label=1.0];
  N2[label=Normal];
  N3[label=Sample];
  N4[label=2.0];
  N5[label="+"];
  N6[label=3.0];
  N7[label="*"];
  N8[label=Query];
  N0 -> N2;
  N1 -> N2;
  N2 -> N3;
  N3 -> N5;
  N4 -> N5;
  N5 -> N7;
  N6 -> N7;
  N7 -> N8;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        # We do this by:
        # * the single assignment rewriter does not fully rewrite
        #   calls to super to their most general form; in particular
        #   it will not rewrite super() to x = [] / super(*x).
        # * the bm_to_bmg rewriter does not rewrite calls to super
        #   into bmg.handle_function.
        # * if the original function has an outer variable __class__ then
        #   we generate a new outer variable with the same name and value.

        # Obtain the random variable for d.foo()
        rv = d.foo()

        # The random variable has a reference to the original *undecorated*
        # D.foo, which has an outer variable __class__. Verify that we
        # correctly recreate that outer variable in the rewritten function:

        bmgast = _bm_function_to_bmg_ast(rv.function, "foo_helper")
        observed = astor.to_source(bmgast)
        expected = """
def foo_helper(bmg, __class__):

    def foo(self):
        a5 = super()
        a1 = bmg.handle_dot_get(a5, 'foo')
        r7 = []
        r10 = {}
        f = bmg.handle_function(a1, r7, r10)
        a14 = [DerivedModel]
        a15 = [self]
        r13 = bmg.handle_addition(a14, a15)
        a6 = super(*r13)
        a2 = bmg.handle_dot_get(a6, 'bar')
        r8 = []
        r11 = {}
        b = bmg.handle_function(a2, r8, r11)
        r3 = bmg.handle_multiplication(f, b)
        return r3
    a4 = bmg.handle_dot_get(bm, 'functional')
    r9 = [foo]
    r12 = {}
    foo = bmg.handle_function(a4, r9, r12)
    return foo
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_subscript_mutations(self) -> None:
        self.maxDiff = None

        observed = BMGInference().to_dot([legal_subscript_mutations()], {})
        expected = """
digraph "graph" {
  N0[label="[11.0,10.0]"];
  N1[label=Dirichlet];
  N2[label=Sample];
  N3[label=Query];
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
}"""
        self.assertEqual(observed.strip(), expected.strip())

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([illegal_subscript_mutation_1()], {})
        # TODO: Better error message
        expected = (
            "Mutating a tensor with a stochastic value "
            + "is not supported in Bean Machine Graph."
        )
        self.assertEqual(expected, str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([illegal_subscript_mutation_2()], {})
        # TODO: Better error message
        expected = "Mutating a stochastic value is not supported in Bean Machine Graph."
        self.assertEqual(expected, str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([illegal_subscript_mutation_3()], {})
        # TODO: Better error message
        expected = (
            "Mutating a collection or tensor with a stochastic index "
            + "is not supported in Bean Machine Graph."
        )
        self.assertEqual(expected, str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([illegal_subscript_mutation_4()], {})
        # TODO: Better error message
        expected = (
            "Mutating a collection or tensor with a stochastic upper index "
            + "is not supported in Bean Machine Graph."
        )
        self.assertEqual(expected, str(ex.exception))

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_dot([illegal_subscript_mutation_5()], {})
        # TODO: Better error message
        expected = (
            "Mutating a collection or tensor with a stochastic step "
            + "is not supported in Bean Machine Graph."
        )
        self.assertEqual(expected, str(ex.exception))
