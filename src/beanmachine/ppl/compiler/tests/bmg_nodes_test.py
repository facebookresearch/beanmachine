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
        n.inputs[0] = r2
        # r1 and r2 now each have one output
        self.assertEqual(r1.outputs.items[n], 1)
        self.assertEqual(r2.outputs.items[n], 1)
