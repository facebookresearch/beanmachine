# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


class BaseModel:
    @bm.random_variable
    def normal(self):
        return Normal(0.0, 1.0)

    @bm.functional
    def foo(self):
        return self.normal() + 1.0


class DerivedModel(BaseModel):
    @bm.functional
    def foo(self):
        # This call is correct but we handle it wrong.
        return super().foo() * 2.0


class CompilerTest(unittest.TestCase):
    def test_super_call(self) -> None:
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
        try:
            D().callout()
        except RuntimeError as e:
            # This exception is expected.
            self.assertEqual("super(): __class__ cell not found", str(e))

        try:
            d = DerivedModel()
            BMGInference().to_dot([d.foo()], {})
        except RuntimeError as e:
            # This exception is wrong.
            self.assertEqual("super(): __class__ cell not found", str(e))

        # TODO: Fix the bug.
