# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.utils.bm_to_bmg import to_python


source1 = """
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli

@sample
def X():
  return Bernoulli(tensor(0.01))

@sample
def Y():
  return Bernoulli(tensor(0.01))

@sample
def Z():
  return Bernoulli(
    1 - exp(log(tensor(0.99)) + X() * log(tensor(0.01)) + Y() * log(tensor(0.01)))
  )
"""

expected1 = """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
bmg = BMGraphBuilder()
from beanmachine.ppl.model.statistical_model import sample
import torch
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli


@memoize
def X():
    a4 = bmg.add_tensor(tensor(0.01))
    r1 = bmg.add_bernoulli(a4)
    return bmg.add_sample(r1)


@memoize
def Y():
    a5 = bmg.add_tensor(tensor(0.01))
    r2 = bmg.add_bernoulli(a5)
    return bmg.add_sample(r2)


@memoize
def Z():
    a7 = bmg.add_real(1)
    a12 = bmg.add_tensor(torch.tensor(-0.010050326585769653))
    a16 = X()
    a18 = bmg.add_tensor(torch.tensor(-4.605170249938965))
    a14 = bmg.add_multiplication(a16, a18)
    a11 = bmg.add_addition(a12, a14)
    a15 = Y()
    a17 = bmg.add_tensor(torch.tensor(-4.605170249938965))
    a13 = bmg.add_multiplication(a15, a17)
    a10 = bmg.add_addition(a11, a13)
    a9 = bmg.add_exp(a10)
    a8 = bmg.add_negate(a9)
    a6 = bmg.add_addition(a7, a8)
    r3 = bmg.add_bernoulli(a6)
    return bmg.add_sample(r3)


X()
Y()
Z()
"""


class CompilerTest(unittest.TestCase):
    def test_to_python(self) -> None:
        """Tests for to_python from bm_to_bmg.py"""
        self.maxDiff = None
        observed1 = to_python(source1)
        self.assertEqual(observed1.strip(), expected1.strip())
