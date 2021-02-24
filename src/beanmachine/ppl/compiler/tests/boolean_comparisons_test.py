# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli


# TODO: An equality comparison involving two Booleans could be
# TODO: turned into an if-then-else.  That is, for Booleans x, y:
# TODO: x == y  -->  if x then y else not y
# TODO: x != y  -->  if x then not y else y
# TODO: x > y   -->  if x then not y else false
# TODO: x >= y  -->  if x then true else not y
# TODO: x < y   -->  if x then false else y
# TODO: x <= y  -->  if x then y else true


@bm.random_variable
def flip(n):
    return Bernoulli(0.5)


@bm.functional
def f1():
    return flip(0) != 1.0


class BooleanComparisonsTest(unittest.TestCase):
    def test_boolean_comparison_errors_1(self) -> None:

        self.maxDiff = None
        bmg = BMGInference()

        # TODO: Raise a better error than a generic ValueError
        # TODO: This error is poorly phrased. "The operator of a query"?
        # TODO: Surely that should be "operand".
        with self.assertRaises(ValueError) as ex:
            bmg.infer([f1()], {}, 10)
        expected = """
The model uses a != operation unsupported by Bean Machine Graph.
The unsupported node is the operator of a Query.
"""
        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())
