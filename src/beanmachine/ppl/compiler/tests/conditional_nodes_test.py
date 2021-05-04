# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BetaNode,
    BinomialNode,
    HalfCauchyNode,
    IfThenElseNode,
    NaturalNode,
    NormalNode,
    PositiveRealNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    PositiveReal,
    Probability,
    Real,
)


class ConditionalNodesTest(unittest.TestCase):
    def test_inf_type_conditional(self) -> None:
        """test_inf_type_conditional"""

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

        # IfThenElse

        # Boolean : Boolean -> Boolean
        # Boolean : Probability -> Probability
        # Boolean : Natural -> Natural
        # Boolean : PositiveReal -> PositiveReal
        # Boolean : Real -> Real
        self.assertEqual(IfThenElseNode(bern, bern, bern).inf_type, Boolean)
        self.assertEqual(IfThenElseNode(bern, bern, beta).inf_type, Probability)
        self.assertEqual(IfThenElseNode(bern, bern, bino).inf_type, Natural)
        self.assertEqual(IfThenElseNode(bern, bern, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, bern, norm).inf_type, Real)

        # Probability : Boolean -> Probability
        # Probability : Probability -> Probability
        # Probability : Natural -> PositiveReal
        # Probability : PositiveReal -> PositiveReal
        # Probability : Real -> Real
        self.assertEqual(IfThenElseNode(bern, beta, bern).inf_type, Probability)
        self.assertEqual(IfThenElseNode(bern, beta, beta).inf_type, Probability)
        self.assertEqual(IfThenElseNode(bern, beta, bino).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, beta, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, beta, norm).inf_type, Real)

        # Natural : Boolean -> Natural
        # Natural : Probability -> PositiveReal
        # Natural : Natural -> Natural
        # Natural : PositiveReal -> PositiveReal
        # Natural : Real -> Real
        self.assertEqual(IfThenElseNode(bern, bino, bern).inf_type, Natural)
        self.assertEqual(IfThenElseNode(bern, bino, beta).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, bino, bino).inf_type, Natural)
        self.assertEqual(IfThenElseNode(bern, bino, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, bino, norm).inf_type, Real)

        # PositiveReal : Boolean -> PositiveReal
        # PositiveReal : Probability -> PositiveReal
        # PositiveReal : Natural -> PositiveReal
        # PositiveReal : PositiveReal -> PositiveReal
        # PositiveReal : Real -> Real
        self.assertEqual(IfThenElseNode(bern, half, bern).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, beta).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, bino).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, norm).inf_type, Real)

        # Real : Boolean -> Real
        # Real : Probability -> Real
        # Real : Natural -> Real
        # Real : PositiveReal -> Real
        # Real : Real -> Real
        self.assertEqual(IfThenElseNode(bern, norm, bern).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, beta).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, bino).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, half).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, norm).inf_type, Real)
