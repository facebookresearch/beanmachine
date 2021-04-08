# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_types.py"""
import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    BooleanMatrix,
    Malformed,
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
    Tensor,
    Zero,
    ZeroMatrix,
    bottom,
    supremum,
    type_meets_requirement,
    type_of_value,
    upper_bound,
)
from beanmachine.ppl.compiler.gen_dot import to_dot
from torch import tensor


class BMGTypesTest(unittest.TestCase):
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
        self.assertEqual(ZeroMatrix(1, 2), type_of_value(tensor([0, 0])))
        self.assertEqual(BooleanMatrix(1, 3), type_of_value(tensor([0, 1, 1])))
        self.assertEqual(BooleanMatrix(1, 2), type_of_value(tensor([1, 1])))
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

    def test_type_meets_requirement(self) -> None:
        """test_type_meets_requirement"""
        self.assertFalse(type_meets_requirement(Natural, Boolean))
        self.assertTrue(type_meets_requirement(Natural, Natural))
        self.assertTrue(type_meets_requirement(Boolean, upper_bound(Natural)))
        self.assertTrue(type_meets_requirement(Natural, upper_bound(Natural)))
        self.assertFalse(type_meets_requirement(PositiveReal, upper_bound(Natural)))

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

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N00[label="0.5:T>=P"];
  N01[label="2.0:T>=N"];
  N02[label="Beta:P>=P"];
  N03[label="Sample:P>=P"];
  N04[label="*:M>=P"];
  N05[label="1.0:T>=OH"];
  N06[label="Normal:R>=R"];
  N07[label="Sample:R>=R"];
  N08[label="Bernoulli:B>=B"];
  N09[label="Sample:B>=B"];
  N10[label="Query:M>=P"];
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

    def test_type_propagation(self) -> None:
        # When we make a mutation to the graph this can cause the type of a node
        # to change, which can then cause the type of an output node to change,
        # and so on. We need to make sure that types propagate correctly.
        #
        # Note that we aggressively compute and store types for performance
        # reasons; since a node's type can depend on the types of its inputs,
        # we could end up traversing large parts of the graph every time we ask
        # a node for its type. Since that operation is common, we want it to be
        # extremely cheap so we compute it once and store the result until it
        # needs to change. Graph mutations, by contrast, are rare and most of the
        # mutations we do will not actually change the type of the outputs so
        # the propagation to outputs will be cheap.
        #
        # To test this though, we'll make some contrived situations that
        # demonstrate the correctness of the propagation.

        bmg = BMGraphBuilder()
        m = bmg.add_real(0.0)
        s = bmg.add_pos_real(1.0)
        norm = bmg.add_normal(m, s)
        ns = bmg.add_sample(norm)
        # ns is a real

        hc = bmg.add_halfcauchy(s)
        hcs = bmg.add_sample(hc)
        # hcs is a positive real

        # Addition requires that its operands have the same type. Let's see
        # what happens when we create an addition and then square it, and
        # then mutate the addition; the type information should propagate
        # to the multiplication.

        add = bmg.add_addition(ns, ns)
        mult = bmg.add_multiplication(add, add)

        self.assertEqual(add.graph_type, Real)
        self.assertEqual(add.inf_type, Real)
        self.assertEqual(mult.graph_type, Real)
        self.assertEqual(mult.inf_type, Real)

        # Now let's mutate the addition so it is malformed.  We should
        # get that the inf type says that there is a mutation that makes
        # this into a real-valued addition, and the graph type of both
        # math operations is malformed.

        add.inputs[0] = hcs

        self.assertEqual(add.graph_type, Malformed)
        self.assertEqual(add.inf_type, Real)
        self.assertEqual(mult.graph_type, Malformed)
        self.assertEqual(mult.inf_type, Real)

        # And now if we mutate further into a multiplication of two
        # positive reals, the problem is fixed:

        add.inputs[1] = hcs

        self.assertEqual(add.graph_type, PositiveReal)
        self.assertEqual(add.inf_type, PositiveReal)
        self.assertEqual(mult.graph_type, PositiveReal)
        self.assertEqual(mult.inf_type, PositiveReal)
