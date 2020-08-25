# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_types.py"""
import unittest

from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    BooleanMatrix,
    Natural,
    PositiveReal,
    Probability,
    Real,
    Tensor,
    meets_requirement,
    supremum,
    type_of_value,
    upper_bound,
)
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from torch import tensor


class BMGTypesTest(unittest.TestCase):
    def test_supremum(self) -> None:
        """test_supremum"""

        self.assertEqual(Boolean, supremum())
        self.assertEqual(Probability, supremum(Probability))
        self.assertEqual(PositiveReal, supremum(Probability, Natural))
        self.assertEqual(Real, supremum(Natural, Probability, Real))
        self.assertEqual(Tensor, supremum(Real, Tensor, Natural, Boolean))

    def test_type_of_value(self) -> None:
        """test_type_of_value"""

        self.assertEqual(Boolean, type_of_value(True))
        self.assertEqual(Boolean, type_of_value(False))
        self.assertEqual(Boolean, type_of_value(0))
        self.assertEqual(Boolean, type_of_value(1))
        self.assertEqual(Boolean, type_of_value(0.0))
        self.assertEqual(Boolean, type_of_value(1.0))
        self.assertEqual(Boolean, type_of_value(tensor(False)))
        self.assertEqual(Boolean, type_of_value(tensor(0)))
        self.assertEqual(Boolean, type_of_value(tensor(1)))
        self.assertEqual(Boolean, type_of_value(tensor(0.0)))
        self.assertEqual(Boolean, type_of_value(tensor(1.0)))
        self.assertEqual(Boolean, type_of_value(tensor([[True]])))
        self.assertEqual(Boolean, type_of_value(tensor([[False]])))
        self.assertEqual(Boolean, type_of_value(tensor([[0]])))
        self.assertEqual(Boolean, type_of_value(tensor([[1]])))
        self.assertEqual(Boolean, type_of_value(tensor([[0.0]])))
        self.assertEqual(Boolean, type_of_value(tensor([[1.0]])))
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
        self.assertEqual(Real, type_of_value(-1.5))
        self.assertEqual(Real, type_of_value(tensor(-1.5)))
        self.assertEqual(Real, type_of_value(tensor([[-1.5]])))
        self.assertEqual(Tensor, type_of_value(tensor([[0, 0]])))

    def test_meets_requirement(self) -> None:
        """test_meets_requirement"""
        self.assertFalse(meets_requirement(Natural, Boolean))
        self.assertTrue(meets_requirement(Natural, Natural))
        self.assertTrue(meets_requirement(Boolean, upper_bound(Natural)))
        self.assertTrue(meets_requirement(Natural, upper_bound(Natural)))
        self.assertFalse(meets_requirement(PositiveReal, upper_bound(Natural)))

    def test_types_in_dot(self) -> None:
        """test_types_in_dot"""
        self.maxDiff = None
        bmg = BMGraphBuilder()
        one = bmg.add_constant(tensor(1.0))
        two = bmg.add_constant(tensor(2.0))
        half = bmg.add_constant(tensor(0.5))
        beta = bmg.add_beta(two, two)
        betas = bmg.add_sample(beta)
        mult = bmg.add_multiplication(half, betas)
        norm = bmg.add_normal(mult, one)
        bern = bmg.add_bernoulli(mult)
        bmg.add_sample(norm)
        bmg.add_sample(bern)
        bmg.add_query(mult)

        observed = bmg.to_dot(True, True, True, True)
        expected = """
digraph "graph" {
  N00[label="1.0:T>=B"];
  N01[label="2.0:T>=N"];
  N02[label="0.5:T>=P"];
  N03[label="Beta:P>=P"];
  N04[label="Sample:P>=P"];
  N05[label="*:M>=P"];
  N06[label="Normal:R>=R"];
  N07[label="Bernoulli:B>=B"];
  N08[label="Sample:R>=R"];
  N09[label="Sample:B>=B"];
  N10[label="Query:M>=P"];
  N00 -> N06[label="sigma:R+"];
  N01 -> N03[label="alpha:R+"];
  N01 -> N03[label="beta:R+"];
  N02 -> N05[label="left:P"];
  N03 -> N04[label="operand:P"];
  N04 -> N05[label="right:P"];
  N05 -> N06[label="mu:R"];
  N05 -> N07[label="probability:P"];
  N05 -> N10[label="operator:P"];
  N06 -> N08[label="operand:R"];
  N07 -> N09[label="operand:B"];
}"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_matrix_types(self) -> None:
        """test_matrix_types"""
        b22 = BooleanMatrix(2, 2)
        b33 = BooleanMatrix(3, 3)

        # Reference equality
        self.assertEqual(b22, BooleanMatrix(2, 2))
        self.assertNotEqual(b22, b33)

        self.assertEqual(b22.short_name, "MB[2,2]")
        self.assertEqual(b22.long_name, "2 x 2 bool matrix")
