# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for bmg_types.py"""
import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import (
    _lookup,
    Boolean,
    BooleanMatrix,
    bottom,
    Natural,
    NaturalMatrix,
    NegativeReal,
    NegativeRealMatrix,
    One,
    OneHotMatrix,
    PositiveReal,
    PositiveRealMatrix,
    Probability,
    ProbabilityMatrix,
    Real,
    RealMatrix,
    SimplexMatrix,
    supremum,
    Tensor,
    type_of_value,
    Zero,
    ZeroMatrix,
)
from beanmachine.ppl.compiler.gen_dot import to_dot
from beanmachine.ppl.model.rv_identifier import RVIdentifier
from torch import tensor


def _rv_id() -> RVIdentifier:
    return RVIdentifier(lambda a, b: a, (1, 1))


class BMGTypesTest(unittest.TestCase):
    def test_lookup_table(self) -> None:
        """Tests _lookup_table for some basic properties"""
        # This is a simple example of "property based testing"
        # Since the search space is finite, we can do it exhaustively
        lookup_table = _lookup()
        keys = lookup_table.keys()
        for key in keys:
            a, b = key
            class_1 = lookup_table[(a, b)]
            class_2 = lookup_table[(b, a)]
            # symmertry
            self.assertEqual(
                class_1(1, 1),
                class_2(1, 1),
                msg="Table breaks symmertry when inputs are ("
                + str(a)
                + ","
                + str(b)
                + ")",
            )

        return

    def test_supremum(self) -> None:
        """test_supremum"""

        # Degenerate case -- supremum of no types is bottom because it
        # is the smallest type that is larger than every type in the
        # empty list.
        self.assertEqual(bottom, supremum())

        # Supremum of one type is that type
        self.assertEqual(Probability, supremum(Probability))

        # A few cases for single-valued types.
        self.assertEqual(PositiveReal, supremum(Probability, Natural))
        self.assertEqual(Real, supremum(Natural, Probability, Real))
        self.assertEqual(Tensor, supremum(Real, Tensor, Natural, Boolean))
        self.assertEqual(Real, supremum(NegativeReal, PositiveReal))
        self.assertEqual(Boolean, supremum(One, Zero))

        # Supremum of any two types with different matrix dimensions is Tensor
        self.assertEqual(Tensor, supremum(RealMatrix(1, 2), RealMatrix(2, 1)))

        # A few cases for matrices

        self.assertEqual(
            ProbabilityMatrix(1, 2), supremum(BooleanMatrix(1, 2), SimplexMatrix(1, 2))
        )

        self.assertEqual(
            PositiveRealMatrix(1, 2), supremum(NaturalMatrix(1, 2), SimplexMatrix(1, 2))
        )

    def test_type_of_value(self) -> None:
        """test_type_of_value"""

        self.assertEqual(One, type_of_value(True))
        self.assertEqual(Zero, type_of_value(False))
        self.assertEqual(Zero, type_of_value(0))
        self.assertEqual(One, type_of_value(1))
        self.assertEqual(Zero, type_of_value(0.0))
        self.assertEqual(One, type_of_value(1.0))
        self.assertEqual(Zero, type_of_value(tensor(False)))
        self.assertEqual(Zero, type_of_value(tensor(0)))
        self.assertEqual(One, type_of_value(tensor(1)))
        self.assertEqual(Zero, type_of_value(tensor(0.0)))
        self.assertEqual(One, type_of_value(tensor(1.0)))
        self.assertEqual(One, type_of_value(tensor([[True]])))
        self.assertEqual(Zero, type_of_value(tensor([[False]])))
        self.assertEqual(Zero, type_of_value(tensor([[0]])))
        self.assertEqual(One, type_of_value(tensor([[1]])))
        self.assertEqual(Zero, type_of_value(tensor([[0.0]])))
        self.assertEqual(One, type_of_value(tensor([[1.0]])))
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
        self.assertEqual(NegativeReal, type_of_value(-1.5))
        self.assertEqual(NegativeReal, type_of_value(tensor(-1.5)))
        self.assertEqual(NegativeReal, type_of_value(tensor([[-1.5]])))

        # 1-d tensor is matrix
        # Tensors are row-major in torch but column-major in BMG. We give
        # the BMG type of the tensor as though it were column-major.
        # This is treated as if it were [[0],[0]], a 2-column 1-row tensor
        # because that's what we're going to emit into BMG.
        self.assertEqual(ZeroMatrix(2, 1), type_of_value(tensor([0, 0])))
        self.assertEqual(BooleanMatrix(3, 1), type_of_value(tensor([0, 1, 1])))
        self.assertEqual(BooleanMatrix(2, 1), type_of_value(tensor([1, 1])))
        # 2-d tensor is matrix
        self.assertEqual(OneHotMatrix(2, 2), type_of_value(tensor([[1, 0], [1, 0]])))
        self.assertEqual(BooleanMatrix(2, 2), type_of_value(tensor([[1, 1], [1, 0]])))
        self.assertEqual(NaturalMatrix(2, 2), type_of_value(tensor([[1, 3], [1, 0]])))
        self.assertEqual(
            SimplexMatrix(2, 2), type_of_value(tensor([[0.5, 0.5], [0.5, 0.5]]))
        )
        self.assertEqual(
            ProbabilityMatrix(2, 2), type_of_value(tensor([[0.75, 0.5], [0.5, 0.5]]))
        )
        self.assertEqual(
            PositiveRealMatrix(2, 2), type_of_value(tensor([[1.75, 0.5], [0.5, 0.5]]))
        )
        self.assertEqual(
            RealMatrix(2, 2), type_of_value(tensor([[1.75, 0.5], [0.5, -0.5]]))
        )
        self.assertEqual(
            NegativeRealMatrix(2, 2),
            type_of_value(tensor([[-1.75, -0.5], [-0.5, -0.5]])),
        )
        # 3-d tensor is Tensor
        self.assertEqual(
            Tensor, type_of_value(tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
        )
        # Empty tensor is Tensor
        self.assertEqual(Tensor, type_of_value(tensor([])))

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
        bmg.add_query(mult, _rv_id())

        observed = to_dot(
            bmg,
            node_types=True,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N00[label="0.5:P"];
  N01[label="2.0:N"];
  N02[label="Beta:P"];
  N03[label="Sample:P"];
  N04[label="*:P"];
  N05[label="1.0:OH"];
  N06[label="Normal:R"];
  N07[label="Sample:R"];
  N08[label="Bernoulli:B"];
  N09[label="Sample:B"];
  N10[label="Query:P"];
  N00 -> N04[label="left:P"];
  N01 -> N02[label="alpha:R+"];
  N01 -> N02[label="beta:R+"];
  N02 -> N03[label="operand:P"];
  N03 -> N04[label="right:P"];
  N04 -> N06[label="mu:R"];
  N04 -> N08[label="probability:P"];
  N04 -> N10[label="operator:any"];
  N05 -> N06[label="sigma:R+"];
  N06 -> N07[label="operand:R"];
  N08 -> N09[label="operand:B"];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_matrix_types(self) -> None:
        """test_matrix_types"""
        b22 = BooleanMatrix(2, 2)
        b33 = BooleanMatrix(3, 3)

        # Reference equality
        self.assertEqual(b22, BooleanMatrix(2, 2))
        self.assertNotEqual(b22, b33)

        self.assertEqual(b22.short_name, "MB[2,2]")
        self.assertEqual(b22.long_name, "2 x 2 bool matrix")
