# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_to_bmg.py"""
import unittest

from beanmachine.ppl.utils.bm_to_bmg import to_python


source1 = """
from beanmachine.ppl.model.statistical_model import sample
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
from beanmachine.ppl.model.statistical_model import sample
from torch import exp, log, tensor
from torch.distributions.bernoulli import Bernoulli


@sample
def X():
    a4 = tensor(0.01)
    r1 = Bernoulli(a4)
    return r1


@sample
def Y():
    a5 = tensor(0.01)
    r2 = Bernoulli(a5)
    return r2


@sample
def Z():
    a7 = 1
    a12 = torch.tensor(-0.010050326585769653)
    a16 = X()
    a18 = torch.tensor(-4.605170249938965)
    a14 = a16 * a18
    a11 = a12 + a14
    a15 = Y()
    a17 = torch.tensor(-4.605170249938965)
    a13 = a15 * a17
    a10 = a11 + a13
    a9 = exp(a10)
    a8 = -a9
    a6 = a7 + a8
    r3 = Bernoulli(a6)
    return r3
"""


class CompilerTest(unittest.TestCase):
    def test_to_python(self) -> None:
        """Tests for to_python from bm_to_bmg.py"""
        self.maxDiff = None
        observed1 = to_python(source1)
        self.assertEqual(observed1.strip(), expected1.strip())
