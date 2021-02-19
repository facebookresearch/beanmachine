# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Normal


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.random_variable
def norm(n):
    return Normal(0, 1)


@bm.functional
def do_it():
    return norm(flip())


class BMGBadModelsTest(unittest.TestCase):
    def test_bmg_inference_error_reporting(self):

        with self.assertRaises(TypeError) as ex:
            BMGInference().infer(123, {}, 10)
        self.assertEqual(
            str(ex.exception),
            "Parameter 'queries' is required to be a list but is of type int.",
        )
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([], 123, 10)
        self.assertEqual(
            str(ex.exception),
            "Parameter 'observations' is required to be a dictionary but is of type int.",
        )

        # Should be flip():
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([flip], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "A query is required to be a random variable but is of type function.",
        )

        # Should be flip():
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([flip()], {flip: tensor(True)}, 10)
        self.assertEqual(
            str(ex.exception),
            "An observation is required to be a random variable but is of type function.",
        )

        # Should be a tensor
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([flip()], {flip(): 123.0}, 10)
        self.assertEqual(
            str(ex.exception),
            "An observed value is required to be a tensor but is of type float.",
        )

        # You can't make inferences on rv-of-rv
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([norm(flip())], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "The arguments to a query must not be random variables.",
        )

        # You can't make inferences on rv-of-rv
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([flip()], {norm(flip()): tensor(123)}, 10)
        self.assertEqual(
            str(ex.exception),
            "The arguments to an observation must not be random variables.",
        )

        # Observations must be of random variables, not
        # functionals
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([flip()], {do_it(): tensor(123)}, 10)
        self.assertEqual(
            str(ex.exception),
            "An observation must observe a random_variable, not a functional.",
        )

        # TODO: Verify that a query that returns a non-node fails at runtime.
        # TODO: Verify we handle correctly the case where a queried value is
        # a constant, because that is not directly supported by BMG but
        # it would be nice to have.
        # TODO: Verify that an RV that returns an unsupported distribution
        # is an error.
