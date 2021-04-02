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
    LogSumExpNode,
    MultiAdditionNode,
    NaturalNode,
    NegativeRealNode,
    NormalNode,
    PositiveRealNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
    StudentTNode,
    TensorNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    NegativeReal,
    One,
    PositiveReal,
    Probability,
    Real,
    Tensor as BMGTensor,
    Zero,
)
from torch import Size


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
        neg = NegativeRealNode(-1.5)
        real = RealNode(-1.5)
        nat = NaturalNode(2)

        self.assertEqual(b.inf_type, One)
        self.assertEqual(bf.inf_type, Zero)
        self.assertEqual(prob.inf_type, Probability)
        self.assertEqual(pos.inf_type, PositiveReal)
        self.assertEqual(neg.inf_type, NegativeReal)
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

        # Tensors
        # TODO: See notes in TensorNode class for how we should improve this.
        self.assertEqual(TensorNode([half, norm], Size([2])).inf_type, BMGTensor)

        # LogSumExp is always real.
        self.assertEqual(LogSumExpNode([half, norm]).inf_type, Real)

        self.assertEqual(MultiAdditionNode([half, norm, beta]).inf_type, Real)

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
        neg = NegativeRealNode(-1.5)
        real = RealNode(-1.5)
        nat = NaturalNode(2)

        self.assertEqual(b.requirements, [])
        self.assertEqual(prob.requirements, [])
        self.assertEqual(pos.requirements, [])
        self.assertEqual(neg.requirements, [])
        self.assertEqual(real.requirements, [])
        self.assertEqual(nat.requirements, [])

        # Distributions

        bern = BernoulliNode(prob)
        beta = BetaNode(pos, pos)

        self.assertEqual(bern.requirements, [Probability])
        self.assertEqual(beta.requirements, [PositiveReal, PositiveReal])
        self.assertEqual(BinomialNode(nat, prob).requirements, [Natural, Probability])
        self.assertEqual(GammaNode(pos, pos).requirements, [PositiveReal, PositiveReal])
        self.assertEqual(Chi2Node(pos).requirements, [PositiveReal])
        self.assertEqual(HalfCauchyNode(pos).requirements, [PositiveReal])
        self.assertEqual(NormalNode(neg, pos).requirements, [Real, PositiveReal])
        self.assertEqual(NormalNode(real, pos).requirements, [Real, PositiveReal])
        self.assertEqual(
            StudentTNode(pos, pos, pos).requirements, [PositiveReal, Real, PositiveReal]
        )

        # Tensors
        # TODO: See notes in TensorNode class for how we should improve this.
        self.assertEqual(
            TensorNode([SampleNode(bern), SampleNode(beta)], Size([2])).requirements,
            [Boolean, Probability],
        )

        # LogSumExp requires the dimension be natural and the values be
        # real, positive real, or negative real.

        self.assertEqual(LogSumExpNode([real, pos]).requirements, [Real, Real])
        self.assertEqual(
            LogSumExpNode([pos, nat]).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            LogSumExpNode([neg, neg]).requirements, [NegativeReal, NegativeReal]
        )

    def test_inputs_and_outputs(self) -> None:
        # We must maintain the invariant that the output set and the
        # input set of every node are consistent even when the graph
        # is edited.

        r1 = RealNode(1.0)
        self.assertEqual(len(r1.outputs.items), 0)
        n = NormalNode(r1, r1)
        # r1 has two outputs, both equal to n
        self.assertEqual(r1.outputs.items[n], 2)
        r2 = RealNode(2.0)
        n.sigma = r2
        # r1 and r2 now each have one output
        self.assertEqual(r1.outputs.items[n], 1)
        self.assertEqual(r2.outputs.items[n], 1)
