# Copyright (c) Facebook, Inc. and its affiliates.
"""A basic unit test for the Python interface of the BMG C++ Graph.infer method"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli


@bm.functional
def c():
    return tensor(2.5)


@bm.functional
def c2():
    return tensor([1.5, -2.5])


@bm.random_variable
def flip():
    return Bernoulli(0.5)


@bm.functional
def flip2():
    return flip()


@bm.functional
def flip3():
    return flip() + 0


@bm.functional
def flip4():
    return 0 + flip()


@bm.functional
def always_false_1():
    return 1 < flip()


@bm.functional
def always_false_2():
    return flip() < 0


@bm.functional
def invalid_tensor_1():
    return tensor([])


@bm.functional
def invalid_tensor_2():
    return tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])


class BMGInferInterfaceTest(unittest.TestCase):
    def test_infer_interface_constant_functional(self) -> None:
        self.maxDiff = None

        samples = BMGInference().infer([c(), c2()], {}, 1)
        observed = samples[c()]
        expected = "tensor([[2.5000]])"
        self.assertEqual(expected.strip(), str(observed).strip())
        observed = samples[c2()]
        expected = "tensor([[[ 1.5000, -2.5000]]], dtype=torch.float64)"
        self.assertEqual(expected.strip(), str(observed).strip())

    def test_infer_interface_redundant_functionals_1(self) -> None:
        self.maxDiff = None

        samples = BMGInference().infer([flip(), flip2()], {}, 10)
        f = samples[flip()]
        f2 = samples[flip2()]
        self.assertEqual(str(f), str(f2))

        samples = BMGInference().infer([always_false_1(), always_false_2()], {}, 2)
        af1 = samples[always_false_1()]
        af2 = samples[always_false_2()]
        expected = "tensor([[False, False]])"
        self.assertEqual(expected, str(af1))
        self.assertEqual(expected, str(af2))

    def test_infer_interface_redundant_functionals_2(self) -> None:
        self.maxDiff = None

        samples = BMGInference().infer([flip3(), flip4()], {}, 10)
        f3 = samples[flip3()]
        f4 = samples[flip4()]
        self.assertEqual(str(f3), str(f4))
