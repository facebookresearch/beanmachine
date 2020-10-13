# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

from beanmachine.ppl.compiler.bmg_nodes import (
    BernoulliNode,
    BetaNode,
    BinomialNode,
    BooleanNode,
    Chi2Node,
    FlatNode,
    GammaNode,
    HalfCauchyNode,
    NaturalNode,
    NormalNode,
    PositiveRealNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
    StudentTNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    NegativeReal,
    One,
    PositiveReal,
    Probability,
    Real,
    Zero,
)


class BMGNodesTest(unittest.TestCase):
    def test_inf_type(self) -> None:
        """test_inf_type"""

        # The infimum type of a node is the *smallest* type that the
        # node can be converted to.

        # Constants

        b = BooleanNode(True)
        bf = BooleanNode(False)
        prob = ProbabilityNode(0.5)
        pos = PositiveRealNode(1.5)
        real = RealNode(-1.5)
        nat = NaturalNode(2)

        self.assertEqual(b.inf_type, One)
        self.assertEqual(bf.inf_type, Zero)
        self.assertEqual(prob.inf_type, Probability)
        self.assertEqual(pos.inf_type, PositiveReal)
        self.assertEqual(real.inf_type, NegativeReal)
        self.assertEqual(nat.inf_type, Natural)

        # Constant infimum type depends on the value,
        # not the type of the container.

        self.assertEqual(ProbabilityNode(1.0).inf_type, One)
        self.assertEqual(NaturalNode(1).inf_type, One)
        self.assertEqual(PositiveRealNode(2.0).inf_type, Natural)
        self.assertEqual(RealNode(2.5).inf_type, PositiveReal)

        # Distributions

        bern = SampleNode(BernoulliNode(prob))
        beta = SampleNode(BetaNode(pos, pos))
        bino = SampleNode(BinomialNode(nat, prob))
        flat = SampleNode(FlatNode())
        gamm = SampleNode(GammaNode(pos, pos))
        chi2 = SampleNode(Chi2Node(pos))
        half = SampleNode(HalfCauchyNode(pos))
        norm = SampleNode(NormalNode(real, pos))
        stut = SampleNode(StudentTNode(pos, pos, pos))

        self.assertEqual(bern.inf_type, Boolean)
        self.assertEqual(beta.inf_type, Probability)
        self.assertEqual(bino.inf_type, Natural)
        self.assertEqual(flat.inf_type, Probability)
        self.assertEqual(chi2.inf_type, PositiveReal)
        self.assertEqual(gamm.inf_type, PositiveReal)
        self.assertEqual(half.inf_type, PositiveReal)
        self.assertEqual(norm.inf_type, Real)
        self.assertEqual(stut.inf_type, Real)

    def test_requirements(self) -> None:
        """test_requirements"""

        # Every node can list the *exact* types that it will accept
        # for each input. In some cases this is independent of inputs;
        # a Bernoulli node accepts only a Probability. In some cases
        # this depends on inputs; a multiplication node with inputs
        # Probability and PositiveReal requires that the Probability
        # be converted via a ToPositiveReal node.

        # TODO: These tests only exercise nodes that we can actually
        # convert to BMG nodes right now. As we add more supported
        # node types to BMG, add tests here that reflect the BMG
        # type system rules.

        # Constants

        b = BooleanNode(True)
        prob = ProbabilityNode(0.5)
        pos = PositiveRealNode(1.5)
        real = RealNode(-1.5)
        nat = NaturalNode(2)

        self.assertEqual(b.requirements, [])
        self.assertEqual(prob.requirements, [])
        self.assertEqual(pos.requirements, [])
        self.assertEqual(real.requirements, [])
        self.assertEqual(nat.requirements, [])

        # Distributions

        self.assertEqual(BernoulliNode(prob).requirements, [Probability])
        self.assertEqual(BetaNode(pos, pos).requirements, [PositiveReal, PositiveReal])
        self.assertEqual(BinomialNode(nat, prob).requirements, [Natural, Probability])
        self.assertEqual(GammaNode(pos, pos).requirements, [PositiveReal, PositiveReal])
        self.assertEqual(Chi2Node(pos).requirements, [PositiveReal])
        self.assertEqual(HalfCauchyNode(pos).requirements, [PositiveReal])
        self.assertEqual(NormalNode(real, pos).requirements, [Real, PositiveReal])
        self.assertEqual(
            StudentTNode(pos, pos, pos).requirements, [PositiveReal, Real, PositiveReal]
        )
