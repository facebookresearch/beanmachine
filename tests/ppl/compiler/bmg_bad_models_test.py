# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
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


@bm.functional
def missing_tensor_instance_function():
    # What happens if we call a function on a tensor instance
    # that does not exist at all?
    return norm(1).not_a_real_function()


@bm.functional
def unsupported_tensor_instance_function_1():
    # Tensor instance function exists but we do not handle it.
    return norm(1).arccos()


@bm.functional
def unsupported_tensor_instance_function_2():
    # Same as above but called via Tensor:
    return torch.Tensor.arccos(norm(1))


@bm.functional
def unsupported_tensor_instance_function_3():
    # Regular receiver, stochastic argument:
    return torch.tensor(7.0).dot(norm(1))


@bm.functional
def unsupported_torch_function():
    # Same as above but called via torch:
    return torch.arccos(norm(1))


@bm.functional
def unsupported_torch_submodule_function():
    # What if we call an unsupported function in submodule of torch?
    return torch.special.erf(norm(1))


@bm.functional
def missing_distribution_function():
    # What happens if we try to get a nonsensical attr from a
    # stochastic distribution?
    return Normal(norm(1), 1.0).no_such_function()


@bm.functional
def unsupported_distribution_function():
    return Normal(norm(1), 1.0).entropy()


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

    def test_bad_tensor_operations(self) -> None:

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_tensor_instance_function_1()], {}, 1)
        expected = """
Function arccos is not supported by Bean Machine Graph.
       """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_tensor_instance_function_2()], {}, 1)
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_torch_function()], {}, 1)
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        expected = """
Function dot is not supported by Bean Machine Graph.
        """

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_tensor_instance_function_3()], {}, 1)
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        # I have no idea why torch gives the name of torch.special.erf as
        # "special_erf" rather than "erf", but it does.
        expected = """
Function special_erf is not supported by Bean Machine Graph.
        """

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_torch_submodule_function()], {}, 1)
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([missing_tensor_instance_function()], {}, 1)
        expected = """
Function not_a_real_function is not supported by Bean Machine Graph.
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([missing_distribution_function()], {}, 1)
        expected = """
Function no_such_function is not supported by Bean Machine Graph.
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())

        with self.assertRaises(ValueError) as ex:
            BMGInference().infer([unsupported_distribution_function()], {}, 1)
        expected = """
Function entropy is not supported by Bean Machine Graph.
        """
        self.assertEqual(expected.strip(), str(ex.exception).strip())
