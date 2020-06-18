# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_types.py"""
import unittest

from beanmachine.ppl.compiler.bmg_types import (
    Natural,
    PositiveReal,
    Probability,
    supremum,
)
from torch import Tensor


class BMGTypesTest(unittest.TestCase):
    def test_supremum(self) -> None:
        """test_supremum"""

        self.assertEqual(bool, supremum())
        self.assertEqual(Probability, supremum(Probability))
        self.assertEqual(PositiveReal, supremum(Probability, Natural))
        self.assertEqual(float, supremum(Natural, Probability, float))
        self.assertEqual(Tensor, supremum(float, Tensor, Natural, bool))
