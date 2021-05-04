# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

from beanmachine.ppl.compiler.bmg_nodes import (
    AdditionNode,
    BernoulliNode,
    BetaNode,
    BinomialNode,
    HalfCauchyNode,
    MultiplicationNode,
    NaturalNode,
    NegateNode,
    NormalNode,
    PositiveRealNode,
    PowerNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    NegativeReal,
    PositiveReal,
    Probability,
    Real,
)


class BMGNodesTest(unittest.TestCase):
    def test_inf_type_binary(self) -> None:
        """test_inf_type_binary"""

        # The infimum type of a node is the *smallest* type that the
        # node can be converted to.

        prob = ProbabilityNode(0.5)
        pos = PositiveRealNode(1.5)
        real = RealNode(-1.5)
        nat = NaturalNode(2)
        bern = SampleNode(BernoulliNode(prob))
        beta = SampleNode(BetaNode(pos, pos))
        bino = SampleNode(BinomialNode(nat, prob))
        half = SampleNode(HalfCauchyNode(pos))
        norm = SampleNode(NormalNode(real, pos))
        neg = NegateNode(half)

        # Multiplication

        # Do not be confused by the notation here. What we are saying in these
        # comments is "the smallest type that a multiplication of a bool by a
        # probability can possibly be is probability".  We are NOT implying
        # here that BMG supports multiplication of bools by probabilities,
        # because it does not. Remember, the purpose of this code is that it
        # is part of an algorithm that computes *requirements* that must be met
        # to satisfy the BMG type system.
        #
        # What this is really saying is "if we multiplied a bool by a probability,
        # we could meet the requirements of the BMG type system by inserting code
        # that converted the bool to a probability."

        # Boolean x Boolean -> Probability
        # Boolean x Probability -> Probability
        # Boolean x Natural -> PositiveReal
        # Boolean x PositiveReal -> PositiveReal
        # Boolean x Real -> Real
        self.assertEqual(MultiplicationNode(bern, bern).inf_type, Probability)
        self.assertEqual(MultiplicationNode(bern, beta).inf_type, Probability)
        self.assertEqual(MultiplicationNode(bern, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bern, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bern, norm).inf_type, Real)

        # Probability x Boolean -> Probability
        # Probability x Probability -> Probability
        # Probability x Natural -> PositiveReal
        # Probability x PositiveReal -> PositiveReal
        # Probability x Real -> Real
        self.assertEqual(MultiplicationNode(beta, bern).inf_type, Probability)
        self.assertEqual(MultiplicationNode(beta, beta).inf_type, Probability)
        self.assertEqual(MultiplicationNode(beta, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(beta, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(beta, norm).inf_type, Real)

        # Natural x Boolean -> PositiveReal
        # Natural x Probability -> PositiveReal
        # Natural x Natural -> PositiveReal
        # Natural x PositiveReal -> PositiveReal
        # Natural x Real -> Real
        self.assertEqual(MultiplicationNode(bino, bern).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, beta).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, norm).inf_type, Real)

        # PositiveReal x Boolean -> PositiveReal
        # PositiveReal x Probability -> PositiveReal
        # PositiveReal x Natural -> PositiveReal
        # PositiveReal x PositiveReal -> PositiveReal
        # PositiveReal x Real -> Real
        self.assertEqual(MultiplicationNode(half, bern).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, beta).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, norm).inf_type, Real)

        # Real x Boolean -> Real
        # Real x Probability -> Real
        # Real x Natural -> Real
        # Real x PositiveReal -> Real
        # Real x Real -> Real
        self.assertEqual(MultiplicationNode(norm, bern).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, beta).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, bino).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, half).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, norm).inf_type, Real)

        # Addition

        # Boolean + Boolean -> PositiveReal
        # Boolean + Probability -> PositiveReal
        # Boolean + Natural -> PositiveReal
        # Boolean + PositiveReal -> PositiveReal
        # Boolean + NegativeReal -> Real
        # Boolean + Real -> Real
        self.assertEqual(AdditionNode(bern, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, neg).inf_type, Real)
        self.assertEqual(AdditionNode(bern, norm).inf_type, Real)

        # Probability + Boolean -> PositiveReal
        # Probability + Probability -> PositiveReal
        # Probability + Natural -> PositiveReal
        # Probability + PositiveReal -> PositiveReal
        # Probability + NegativeReal -> Real
        # Probability + Real -> Real
        self.assertEqual(AdditionNode(beta, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, neg).inf_type, Real)
        self.assertEqual(AdditionNode(beta, norm).inf_type, Real)

        # Natural + Boolean -> PositiveReal
        # Natural + Probability -> PositiveReal
        # Natural + Natural -> PositiveReal
        # Natural + PositiveReal -> PositiveReal
        # Natural + NegativeReal -> Real
        # Natural + Real -> Real
        self.assertEqual(AdditionNode(bino, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, neg).inf_type, Real)
        self.assertEqual(AdditionNode(bino, norm).inf_type, Real)

        # PositiveReal + Boolean -> PositiveReal
        # PositiveReal + Probability -> PositiveReal
        # PositiveReal + Natural -> PositiveReal
        # PositiveReal + PositiveReal -> PositiveReal
        # PositiveReal + NegativeReal -> Real
        # PositiveReal + Real -> Real
        self.assertEqual(AdditionNode(half, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, neg).inf_type, Real)
        self.assertEqual(AdditionNode(half, norm).inf_type, Real)

        # NegativeReal + Boolean -> Real
        # NegativeReal + Probability -> Real
        # NegativeReal + Natural -> Real
        # NegativeReal + PositiveReal -> Real
        # NegativeReal + NegativeReal -> NegativeReal
        # NegativeReal + Real -> Real
        self.assertEqual(AdditionNode(neg, bern).inf_type, Real)
        self.assertEqual(AdditionNode(neg, beta).inf_type, Real)
        self.assertEqual(AdditionNode(neg, bino).inf_type, Real)
        self.assertEqual(AdditionNode(neg, half).inf_type, Real)
        self.assertEqual(AdditionNode(neg, neg).inf_type, NegativeReal)
        self.assertEqual(AdditionNode(neg, norm).inf_type, Real)

        # Real + Boolean -> Real
        # Real + Probability -> Real
        # Real + Natural -> Real
        # Real + PositiveReal -> Real
        # Real + NegativeReal -> Real
        # Real + Real -> Real
        self.assertEqual(AdditionNode(norm, bern).inf_type, Real)
        self.assertEqual(AdditionNode(norm, beta).inf_type, Real)
        self.assertEqual(AdditionNode(norm, bino).inf_type, Real)
        self.assertEqual(AdditionNode(norm, half).inf_type, Real)
        self.assertEqual(AdditionNode(norm, neg).inf_type, Real)
        self.assertEqual(AdditionNode(norm, norm).inf_type, Real)

        # Power
        # P ** R+  --> P
        # P ** R   --> R+
        # R+ ** R+ --> R+
        # R+ ** R  --> R+
        # R ** R+  --> R
        # R ** R   --> R

        # Base is B; treated as P
        self.assertEqual(PowerNode(bern, bern).inf_type, Probability)
        self.assertEqual(PowerNode(bern, beta).inf_type, Probability)
        self.assertEqual(PowerNode(bern, bino).inf_type, Probability)
        self.assertEqual(PowerNode(bern, half).inf_type, Probability)
        self.assertEqual(PowerNode(bern, norm).inf_type, PositiveReal)

        # Base is P
        self.assertEqual(PowerNode(beta, bern).inf_type, Probability)
        self.assertEqual(PowerNode(beta, beta).inf_type, Probability)
        self.assertEqual(PowerNode(beta, bino).inf_type, Probability)
        self.assertEqual(PowerNode(beta, half).inf_type, Probability)
        self.assertEqual(PowerNode(beta, norm).inf_type, PositiveReal)

        # Base is N; treated as R+
        self.assertEqual(PowerNode(bino, bern).inf_type, PositiveReal)
        self.assertEqual(PowerNode(bino, beta).inf_type, PositiveReal)
        self.assertEqual(PowerNode(bino, bino).inf_type, PositiveReal)
        self.assertEqual(PowerNode(bino, half).inf_type, PositiveReal)
        self.assertEqual(PowerNode(bino, norm).inf_type, PositiveReal)

        # Base is R+
        self.assertEqual(PowerNode(half, bern).inf_type, PositiveReal)
        self.assertEqual(PowerNode(half, beta).inf_type, PositiveReal)
        self.assertEqual(PowerNode(half, bino).inf_type, PositiveReal)
        self.assertEqual(PowerNode(half, half).inf_type, PositiveReal)
        self.assertEqual(PowerNode(half, norm).inf_type, PositiveReal)

        # Base is R
        self.assertEqual(PowerNode(norm, bern).inf_type, Real)
        self.assertEqual(PowerNode(norm, beta).inf_type, Real)
        self.assertEqual(PowerNode(norm, bino).inf_type, Real)
        self.assertEqual(PowerNode(norm, half).inf_type, Real)
        self.assertEqual(PowerNode(norm, norm).inf_type, Real)
