# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_types.py"""
import unittest

from beanmachine.ppl.compiler.bmg_types import (
    Natural,
    PositiveReal,
    Probability,
    supremum,
    type_of_value,
)
from torch import Tensor, tensor


class BMGTypesTest(unittest.TestCase):
    def test_supremum(self) -> None:
        """test_supremum"""

        self.assertEqual(bool, supremum())
        self.assertEqual(Probability, supremum(Probability))
        self.assertEqual(PositiveReal, supremum(Probability, Natural))
        self.assertEqual(float, supremum(Natural, Probability, float))
        self.assertEqual(Tensor, supremum(float, Tensor, Natural, bool))

    def test_type_of_value(self) -> None:
        """test_type_of_value"""

        self.assertEqual(bool, type_of_value(True))
        self.assertEqual(bool, type_of_value(False))
        self.assertEqual(bool, type_of_value(0))
        self.assertEqual(bool, type_of_value(1))
        self.assertEqual(bool, type_of_value(0.0))
        self.assertEqual(bool, type_of_value(1.0))
        self.assertEqual(bool, type_of_value(tensor(True)))
        self.assertEqual(bool, type_of_value(tensor(False)))
        self.assertEqual(bool, type_of_value(tensor(0)))
        self.assertEqual(bool, type_of_value(tensor(1)))
        self.assertEqual(bool, type_of_value(tensor(0.0)))
        self.assertEqual(bool, type_of_value(tensor(1.0)))
        self.assertEqual(bool, type_of_value(tensor([[True]])))
        self.assertEqual(bool, type_of_value(tensor([[False]])))
        self.assertEqual(bool, type_of_value(tensor([[0]])))
        self.assertEqual(bool, type_of_value(tensor([[1]])))
        self.assertEqual(bool, type_of_value(tensor([[0.0]])))
        self.assertEqual(bool, type_of_value(tensor([[1.0]])))
        self.assertEqual(Natural, type_of_value(2))
        self.assertEqual(Natural, type_of_value(2.0))
        self.assertEqual(Natural, type_of_value(tensor(2)))
        self.assertEqual(Natural, type_of_value(tensor(2.0)))
        self.assertEqual(Natural, type_of_value(tensor([[2]])))
        self.assertEqual(Natural, type_of_value(tensor([[2.0]])))
        self.assertEqual(Probability, type_of_value(0.5))
        self.assertEqual(Probability, type_of_value(tensor(0.5)))
        self.assertEqual(Probability, type_of_value(tensor([[0.5]])))
        self.assertEqual(PositiveReal, type_of_value(1.5))
        self.assertEqual(PositiveReal, type_of_value(tensor(1.5)))
        self.assertEqual(PositiveReal, type_of_value(tensor([[1.5]])))
        self.assertEqual(float, type_of_value(-1.5))
        self.assertEqual(float, type_of_value(tensor(-1.5)))
        self.assertEqual(float, type_of_value(tensor([[-1.5]])))
        self.assertEqual(Tensor, type_of_value(tensor([[0, 0]])))
