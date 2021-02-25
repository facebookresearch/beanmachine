# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor

# Bean Machine allows queries on functionals that return constants;
# BMG does not. It would be nice though if a BM model that queried
# a constant worked when using BMGInference the same way that it
# does with other inference engines, for several reasons:
#
# (1) consistency of behaviour across inference engines
# (2) testing optimizations; if an optimization ends up producing
#     a constant, it's nice to be able to query that functional
#     and see that it does indeed produce a constant.
# (3) possible future error reporting; it would be nice to warn the
#     user that they are querying a constant because this could be
#     a bug in their model.
# (4) model development and debugging; a user might make a dummy functional
#     that just returns a constant now, intending to replace it with an
#     actual function later.  Or might force a functional to produce a
#     particular value to see how the model behaves in that case.
#
# This test verifies that we can query a constant functional.


@bm.functional
def c():
    return tensor(1.0)


class BMGQueryTest(unittest.TestCase):
    def disabled_test_constant_functional(self) -> None:

        # TODO: This test is disabled because there is a bug
        # in the type checker which crashes when it tries
        # to verify the type of a queried one-hot tensor.
        # Fix that bug and then re-enable this test

        self.maxDiff = None

        observed = BMGInference().to_dot([c()], {})
        expected = """ """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp([c()], {})
        expected = """ """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python([c()], {})
        expected = """ """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().infer([c()], {}, 1)[c()]
        expected = """ """
        self.assertEqual(str(expected).strip(), observed.strip())
