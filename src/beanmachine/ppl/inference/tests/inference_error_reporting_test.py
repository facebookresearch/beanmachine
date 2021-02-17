# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from torch import tensor


@bm.random_variable
def f():
    pass


@bm.random_variable
def g(n):
    pass


@bm.functional
def h():
    return 123  # BAD; needs to be a tensor


class InferenceErrorReportingTest(unittest.TestCase):
    def test_inference_error_reporting(self):
        mh = bm.SingleSiteAncestralMetropolisHastings()
        with self.assertRaises(TypeError) as ex:
            mh.infer(None, {}, 10)
        self.assertEqual(
            str(ex.exception),
            "Parameter 'queries' is required to be a list but is of type NoneType.",
        )
        with self.assertRaises(TypeError) as ex:
            mh.infer([], 123, 10)
        self.assertEqual(
            str(ex.exception),
            "Parameter 'observations' is required to be a dictionary but is of type int.",
        )

        # Should be f():
        with self.assertRaises(TypeError) as ex:
            mh.infer([f], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "A query is required to be a random variable but is of type function.",
        )

        # Should be f():
        with self.assertRaises(TypeError) as ex:
            mh.infer([f()], {f: tensor(True)}, 10)
        self.assertEqual(
            str(ex.exception),
            "An observation is required to be a random variable but is of type function.",
        )

        # Should be a tensor
        with self.assertRaises(TypeError) as ex:
            mh.infer([f()], {f(): 123.0}, 10)
        self.assertEqual(
            str(ex.exception),
            "An observed value is required to be a tensor but is of type float.",
        )

        # You can't make inferences on rv-of-rv
        with self.assertRaises(TypeError) as ex:
            mh.infer([g(f())], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "The arguments to a query must not be random variables.",
        )

        # You can't make inferences on rv-of-rv
        with self.assertRaises(TypeError) as ex:
            mh.infer([f()], {g(f()): tensor(123)}, 10)
        self.assertEqual(
            str(ex.exception),
            "The arguments to an observation must not be random variables.",
        )

        # SSAMH requires that observations must be of random variables, not
        # functionals
        with self.assertRaises(TypeError) as ex:
            mh.infer([f()], {h(): tensor(123)}, 10)
        self.assertEqual(
            str(ex.exception),
            "An observation must observe a random_variable, not a functional.",
        )

        # A functional is required to return a tensor.
        with self.assertRaises(TypeError) as ex:
            mh.infer([h()], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "The value returned by a queried function must be a tensor.",
        )
