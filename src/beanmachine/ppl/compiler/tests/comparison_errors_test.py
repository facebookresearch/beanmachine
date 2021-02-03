# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli, Normal, StudentT


# Comparisons involving a graph node have no representation in
# BMG and should produce an error.
#
# Tests which show how the comparison operators are lowered into
# a more fundamental form are in comparison_rewriting_test.py.


@bm.random_variable
def normal():
    return Normal(0.0, 1.0)


@bm.random_variable
def t():
    gt = normal() > 2.0
    gte = normal() >= 2.0
    lt = normal() < 2.0
    lte = normal() <= 2.0
    s = 1.0 + gt + gte + lt + lte
    return StudentT(1.0, s, 1.0)


# TODO: An equality comparison involving two Booleans could be
# TODO: turned into an if-then-else.  That is, for Booleans x, y:
# TODO: x == y  -->  if x then y else not y
# TODO: x != y  -->  if x then not y else y


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def f():
    return flip() != 1.0


class ComparisonErrorsTest(unittest.TestCase):
    def test_comparison_errors_1(self) -> None:

        self.maxDiff = None
        bmg = BMGInference()

        # TODO: Raise a better error than a generic ValueError
        with self.assertRaises(ValueError) as ex:
            bmg.infer([t()], {}, 10)
        expected = """
The model uses a < operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a <= operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a > operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
The model uses a >= operation unsupported by Bean Machine Graph.
The unsupported node is the right of a +.
"""
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())

    def test_comparison_errors_2(self) -> None:

        self.maxDiff = None
        bmg = BMGInference()

        # TODO: Raise a better error than a generic ValueError
        # TODO: This error is poorly phrased. "The operator of a query"?
        # TODO: Surely that should be "operand".
        with self.assertRaises(ValueError) as ex:
            bmg.infer([f()], {}, 10)
        expected = """
The model uses a != operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
"""
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())
