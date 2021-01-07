# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from torch import tensor


@bm.random_variable
def f():
    pass


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
