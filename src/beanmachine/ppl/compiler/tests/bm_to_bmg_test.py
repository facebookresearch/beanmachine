# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

import astor
import beanmachine.ppl as bm
from beanmachine.ppl.compiler.bm_to_bmg import _bm_function_to_bmg_ast
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


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
        # * we generate an outer variable __class__ which is initialized
        #   to the same value as the original function's outer variable
        #   __class__, if it has one, None otherwise.

        bmgast, _ = _bm_function_to_bmg_ast(d.foo, "foo_helper")
        observed = astor.to_source(bmgast)
        expected = """
def foo_helper(bmg, __class__):

    def foo(self):
        a4 = super()
        a1 = bmg.handle_dot_get(a4, 'foo')
        r6 = []
        r8 = {}
        f = bmg.handle_function(a1, r6, r8)
        a11 = [DerivedModel]
        a12 = [self]
        r10 = bmg.handle_addition(a11, a12)
        a5 = super(*r10)
        a2 = bmg.handle_dot_get(a5, 'bar')
        r7 = []
        r9 = {}
        b = bmg.handle_function(a2, r7, r9)
        r3 = bmg.handle_multiplication(f, b)
        return r3
    return foo
"""
        self.assertEqual(observed.strip(), expected.strip())
