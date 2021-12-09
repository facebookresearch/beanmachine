# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Cauchy, Normal


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.random_variable
def norm(n):
    return Normal(0, 1)


@bm.functional
def do_it():
    return norm(flip())


@bm.functional
def bad_functional():
    return 123


@bm.random_variable
def no_distribution_rv():
    return 123


@bm.random_variable
def unsupported_distribution_rv():
    return Cauchy(1.0, 2.0)


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

        # A functional must always return a value that can be represented
        # in the graph.
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([bad_functional()], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "A functional must return a tensor.",
        )

        # TODO: Verify we handle correctly the case where a queried value is
        # a constant, because that is not directly supported by BMG but
        # it would be nice to have.

        # An rv must return a distribution.
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([no_distribution_rv()], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "A random_variable is required to return a distribution.",
        )

        # An rv must return a supported distribution.
        with self.assertRaises(TypeError) as ex:
            BMGInference().infer([unsupported_distribution_rv()], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "Distribution 'Cauchy' is not supported by Bean Machine Graph.",
        )
