# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BetaNode,
    BinomialNode,
    ComplementNode,
    ExpNode,
    HalfCauchyNode,
    LogNode,
    NaturalNode,
    NegateNode,
    NegativeLogNode,
    NormalNode,
    PositiveRealNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
    ToPositiveRealNode,
    ToRealNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
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

        self.assertEqual(bern.inf_type, Boolean)
        self.assertEqual(beta.inf_type, Probability)
        self.assertEqual(bino.inf_type, Natural)
        self.assertEqual(half.inf_type, PositiveReal)
        self.assertEqual(norm.inf_type, Real)

        # Negate
        # - Boolean -> Real
        # - Probability -> Real
        # - Natural -> Real
        # - PositiveReal -> Real
        # - Real -> Real
        self.assertEqual(NegateNode(bern).inf_type, Real)
        self.assertEqual(NegateNode(beta).inf_type, Real)
        self.assertEqual(NegateNode(bino).inf_type, Real)
        self.assertEqual(NegateNode(half).inf_type, Real)
        self.assertEqual(NegateNode(norm).inf_type, Real)

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
        self.assertEqual(ExpNode(bern).inf_type, PositiveReal)
        self.assertEqual(ExpNode(beta).inf_type, PositiveReal)
        self.assertEqual(ExpNode(bino).inf_type, PositiveReal)
        self.assertEqual(ExpNode(half).inf_type, PositiveReal)
        self.assertEqual(ExpNode(norm).inf_type, PositiveReal)

        # Log of anything is Real
        self.assertEqual(LogNode(bern).inf_type, Real)
        self.assertEqual(LogNode(beta).inf_type, Real)
        self.assertEqual(LogNode(bino).inf_type, Real)
        self.assertEqual(LogNode(half).inf_type, Real)

        # Negative Log of probability or smaller is Positive Real;
        # otherwise Real.
        self.assertEqual(NegativeLogNode(bern).inf_type, PositiveReal)
        self.assertEqual(NegativeLogNode(beta).inf_type, PositiveReal)
        self.assertEqual(NegativeLogNode(bino).inf_type, Real)
        self.assertEqual(NegativeLogNode(half).inf_type, Real)

        # To Real
        self.assertEqual(ToRealNode(bern).inf_type, Real)
        self.assertEqual(ToRealNode(beta).inf_type, Real)
        self.assertEqual(ToRealNode(bino).inf_type, Real)
        self.assertEqual(ToRealNode(half).inf_type, Real)
        self.assertEqual(ToRealNode(norm).inf_type, Real)

        # To Positive Real
        self.assertEqual(ToPositiveRealNode(bern).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(beta).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(bino).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(half).inf_type, PositiveReal)

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

        # Negate
        # - Boolean -> Real
        # - Probability -> Real
        # - Natural -> Real
        # - PositiveReal -> Real
        # - Real -> Real
        self.assertEqual(NegateNode(bern).requirements, [Real])
        self.assertEqual(NegateNode(beta).requirements, [Real])
        self.assertEqual(NegateNode(bino).requirements, [Real])
        self.assertEqual(NegateNode(half).requirements, [Real])
        self.assertEqual(NegateNode(norm).requirements, [Real])

        # Complement requires that its operand be probability or Boolean
        self.assertEqual(ComplementNode(bern).requirements, [Boolean])
        self.assertEqual(ComplementNode(beta).requirements, [Probability])

        # Exp requires that its operand be positive real or real.

        self.assertEqual(ExpNode(bern).requirements, [PositiveReal])
        self.assertEqual(ExpNode(beta).requirements, [PositiveReal])
        self.assertEqual(ExpNode(bino).requirements, [PositiveReal])
        self.assertEqual(ExpNode(half).requirements, [PositiveReal])
        self.assertEqual(ExpNode(norm).requirements, [Real])

        # Log requires that its operand be positive real.
        self.assertEqual(LogNode(bern).requirements, [PositiveReal])
        self.assertEqual(LogNode(beta).requirements, [PositiveReal])
        self.assertEqual(LogNode(bino).requirements, [PositiveReal])
        self.assertEqual(LogNode(half).requirements, [PositiveReal])

        # Negative Log requires that its operand be positive real or probability.
        self.assertEqual(NegativeLogNode(bern).requirements, [Probability])
        self.assertEqual(NegativeLogNode(beta).requirements, [Probability])
        self.assertEqual(NegativeLogNode(bino).requirements, [PositiveReal])
        self.assertEqual(NegativeLogNode(half).requirements, [PositiveReal])

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
