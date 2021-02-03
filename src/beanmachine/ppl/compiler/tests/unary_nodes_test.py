# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BetaNode,
    BinomialNode,
    ComplementNode,
    ExpM1Node,
    ExpNode,
    HalfCauchyNode,
    LogNode,
    NaturalNode,
    NegateNode,
    NormalNode,
    PhiNode,
    PositiveRealNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
    ToPositiveRealNode,
    ToProbabilityNode,
    ToRealNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    NegativeReal,
    PositiveReal,
    Probability,
    Real,
    upper_bound,
)


class UnaryNodesTest(unittest.TestCase):
    def test_inf_type_unary(self) -> None:
        """test_inf_type_unary"""

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

        self.assertEqual(bern.inf_type, Boolean)
        self.assertEqual(beta.inf_type, Probability)
        self.assertEqual(bino.inf_type, Natural)
        self.assertEqual(half.inf_type, PositiveReal)
        self.assertEqual(norm.inf_type, Real)
        self.assertEqual(neg.inf_type, NegativeReal)

        # Negate
        # - Boolean -> Real
        # - Probability -> NegativeReal
        # - Natural -> NegativeReal
        # - PositiveReal -> NegativeReal
        # - TODO: Add a test for NegativeReal -> PositiveReal
        # - Real -> Real
        self.assertEqual(NegateNode(bern).inf_type, NegativeReal)
        self.assertEqual(NegateNode(beta).inf_type, NegativeReal)
        self.assertEqual(NegateNode(bino).inf_type, NegativeReal)
        self.assertEqual(NegateNode(half).inf_type, NegativeReal)
        self.assertEqual(NegateNode(norm).inf_type, Real)
        self.assertEqual(NegateNode(neg).inf_type, PositiveReal)

        # Complement
        # - Boolean -> Boolean
        # - Probability -> Probability
        # Everything else is illegal
        self.assertEqual(ComplementNode(bern).inf_type, Boolean)
        self.assertEqual(ComplementNode(beta).inf_type, Probability)

        # Exp
        # exp Boolean -> PositiveReal
        # exp Probability -> PositiveReal
        # exp Natural -> PositiveReal
        # exp PositiveReal -> PositiveReal
        # exp Real -> PositiveReal
        # exp NegativeReal -> Probability
        self.assertEqual(ExpNode(bern).inf_type, PositiveReal)
        self.assertEqual(ExpNode(beta).inf_type, PositiveReal)
        self.assertEqual(ExpNode(bino).inf_type, PositiveReal)
        self.assertEqual(ExpNode(half).inf_type, PositiveReal)
        self.assertEqual(ExpNode(norm).inf_type, PositiveReal)
        self.assertEqual(ExpNode(neg).inf_type, Probability)

        # ExpM1
        # expm1 Boolean -> PositiveReal
        # expm1 Probability -> PositiveReal
        # expm1 Natural -> PositiveReal
        # expm1 PositiveReal -> PositiveReal
        # expm1 Real -> Real
        # expm1 NegativeReal -> NegativeReal
        self.assertEqual(ExpM1Node(bern).inf_type, PositiveReal)
        self.assertEqual(ExpM1Node(beta).inf_type, PositiveReal)
        self.assertEqual(ExpM1Node(bino).inf_type, PositiveReal)
        self.assertEqual(ExpM1Node(half).inf_type, PositiveReal)
        self.assertEqual(ExpM1Node(norm).inf_type, Real)
        self.assertEqual(ExpM1Node(neg).inf_type, NegativeReal)

        # Log of prob is negative real, otherwise real.
        self.assertEqual(LogNode(bern).inf_type, NegativeReal)
        self.assertEqual(LogNode(beta).inf_type, NegativeReal)
        self.assertEqual(LogNode(bino).inf_type, Real)
        self.assertEqual(LogNode(half).inf_type, Real)

        # Phi of anything is Probability
        self.assertEqual(PhiNode(bern).inf_type, Probability)
        self.assertEqual(PhiNode(beta).inf_type, Probability)
        self.assertEqual(PhiNode(bino).inf_type, Probability)
        self.assertEqual(PhiNode(half).inf_type, Probability)

        # To Real
        self.assertEqual(ToRealNode(bern).inf_type, Real)
        self.assertEqual(ToRealNode(beta).inf_type, Real)
        self.assertEqual(ToRealNode(bino).inf_type, Real)
        self.assertEqual(ToRealNode(half).inf_type, Real)
        self.assertEqual(ToRealNode(norm).inf_type, Real)
        self.assertEqual(ToRealNode(neg).inf_type, Real)

        # To Positive Real
        self.assertEqual(ToPositiveRealNode(bern).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(beta).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(bino).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(half).inf_type, PositiveReal)

        # To Probability
        # Input must be real, positive real, or probability
        self.assertEqual(ToProbabilityNode(norm).inf_type, Probability)
        self.assertEqual(ToProbabilityNode(half).inf_type, Probability)
        self.assertEqual(ToProbabilityNode(beta).inf_type, Probability)

    def test_requirements_unary(self) -> None:
        """test_requirements_unary"""

        # Every node can list the *exact* types that it will accept
        # for each input. In some cases this is independent of inputs;
        # a Bernoulli node accepts only a Probability. In some cases
        # this depends on inputs; a multiplication node with inputs
        # Probability and PositiveReal requires that the Probability
        # be converted via a ToPositiveReal node.

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

        # Negate requires its operand be positive real, negative real or real.
        self.assertEqual(NegateNode(bern).requirements, [PositiveReal])
        self.assertEqual(NegateNode(beta).requirements, [PositiveReal])
        self.assertEqual(NegateNode(bino).requirements, [PositiveReal])
        self.assertEqual(NegateNode(half).requirements, [PositiveReal])
        self.assertEqual(NegateNode(norm).requirements, [Real])
        self.assertEqual(NegateNode(neg).requirements, [NegativeReal])

        # Complement requires that its operand be probability or Boolean
        self.assertEqual(ComplementNode(bern).requirements, [Boolean])
        self.assertEqual(ComplementNode(beta).requirements, [Probability])

        # Exp requires that its operand be negative real, positive real or real.

        self.assertEqual(ExpNode(bern).requirements, [PositiveReal])
        self.assertEqual(ExpNode(beta).requirements, [PositiveReal])
        self.assertEqual(ExpNode(bino).requirements, [PositiveReal])
        self.assertEqual(ExpNode(half).requirements, [PositiveReal])
        self.assertEqual(ExpNode(norm).requirements, [Real])
        self.assertEqual(ExpNode(neg).requirements, [NegativeReal])

        # ExpM1 requires that its operand be negative real, positive real or real.

        self.assertEqual(ExpM1Node(bern).requirements, [PositiveReal])
        self.assertEqual(ExpM1Node(beta).requirements, [PositiveReal])
        self.assertEqual(ExpM1Node(bino).requirements, [PositiveReal])
        self.assertEqual(ExpM1Node(half).requirements, [PositiveReal])
        self.assertEqual(ExpM1Node(norm).requirements, [Real])
        self.assertEqual(ExpM1Node(neg).requirements, [NegativeReal])

        # Log requires that its operand be probability or positive real.
        self.assertEqual(LogNode(bern).requirements, [Probability])
        self.assertEqual(LogNode(beta).requirements, [Probability])
        self.assertEqual(LogNode(bino).requirements, [PositiveReal])
        self.assertEqual(LogNode(half).requirements, [PositiveReal])

        # Phi requires that its operand be real.
        self.assertEqual(PhiNode(bern).requirements, [Real])
        self.assertEqual(PhiNode(beta).requirements, [Real])
        self.assertEqual(PhiNode(bino).requirements, [Real])
        self.assertEqual(PhiNode(half).requirements, [Real])

        # To Real
        self.assertEqual(ToRealNode(bern).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(beta).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(bino).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(half).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(norm).requirements, [upper_bound(Real)])

        # To Positive Real
        self.assertEqual(
            ToPositiveRealNode(bern).requirements, [upper_bound(PositiveReal)]
        )
        self.assertEqual(
            ToPositiveRealNode(beta).requirements, [upper_bound(PositiveReal)]
        )
        self.assertEqual(
            ToPositiveRealNode(bino).requirements, [upper_bound(PositiveReal)]
        )
        self.assertEqual(
            ToPositiveRealNode(half).requirements, [upper_bound(PositiveReal)]
        )
        # To Positive Real is illegal on reals and tensors
