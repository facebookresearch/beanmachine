# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bmg_nodes.py"""
import unittest

from beanmachine.ppl.compiler.bmg_nodes import (
    AdditionNode,
    BernoulliNode,
    BetaNode,
    BinomialNode,
    BooleanNode,
    Chi2Node,
    ExpNode,
    FlatNode,
    GammaNode,
    HalfCauchyNode,
    IfThenElseNode,
    MultiplicationNode,
    NaturalNode,
    NegateNode,
    NormalNode,
    PositiveRealNode,
    PowerNode,
    ProbabilityNode,
    RealNode,
    SampleNode,
    StudentTNode,
    TensorNode,
    ToPositiveRealNode,
    ToRealNode,
    ToTensorNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Natural,
    PositiveReal,
    Probability,
    Real,
    upper_bound,
)
from torch import Tensor, tensor


class ASTToolsTest(unittest.TestCase):
    def test_inf_type(self) -> None:
        """test_inf_type"""

        # The infimum type of a node is the *smallest* type that the
        # node can be converted to.

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
        t = TensorNode(tensor([1.0, 1.0]))

        self.assertEqual(b.inf_type, bool)
        self.assertEqual(prob.inf_type, Probability)
        self.assertEqual(pos.inf_type, PositiveReal)
        self.assertEqual(real.inf_type, Real)
        self.assertEqual(nat.inf_type, Natural)
        self.assertEqual(t.inf_type, Tensor)

        # Constant infimum type depends on the value,
        # not the type of the container.

        self.assertEqual(ProbabilityNode(1.0).inf_type, bool)
        self.assertEqual(NaturalNode(1).inf_type, bool)
        self.assertEqual(PositiveRealNode(2.0).inf_type, Natural)
        self.assertEqual(RealNode(2.5).inf_type, PositiveReal)
        self.assertEqual(TensorNode(tensor(1.5)).inf_type, PositiveReal)

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

        self.assertEqual(bern.inf_type, bool)
        self.assertEqual(beta.inf_type, Probability)
        self.assertEqual(bino.inf_type, Natural)
        self.assertEqual(flat.inf_type, Probability)
        self.assertEqual(chi2.inf_type, PositiveReal)
        self.assertEqual(gamm.inf_type, PositiveReal)
        self.assertEqual(half.inf_type, PositiveReal)
        self.assertEqual(norm.inf_type, Real)
        self.assertEqual(stut.inf_type, Real)

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

        # In fact, a bool multiplied by any type can be converted to an "if-then-else"
        # that chooses between the other operand and a zero of that type, so they
        # are all possible.

        # bool x bool -> bool
        # bool x Probability -> Probability
        # bool x Natural -> Natural
        # bool x PositiveReal -> PositiveReal
        # bool x Real -> Real
        # bool x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(bern, bern).inf_type, bool)
        self.assertEqual(MultiplicationNode(bern, beta).inf_type, Probability)
        self.assertEqual(MultiplicationNode(bern, bino).inf_type, Natural)
        self.assertEqual(MultiplicationNode(bern, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bern, norm).inf_type, Real)
        self.assertEqual(MultiplicationNode(bern, t).inf_type, Tensor)

        # Probability x bool -> Probability
        # Probability x Probability -> Probability
        # Probability x Natural -> PositiveReal
        # Probability x PositiveReal -> PositiveReal
        # Probability x Real -> Real
        # Probability x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(beta, bern).inf_type, Probability)
        self.assertEqual(MultiplicationNode(beta, beta).inf_type, Probability)
        self.assertEqual(MultiplicationNode(beta, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(beta, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(beta, norm).inf_type, Real)
        self.assertEqual(MultiplicationNode(beta, t).inf_type, Tensor)

        # Natural x bool -> Natural
        # Natural x Probability -> PositiveReal
        # Natural x Natural -> PositiveReal
        # Natural x PositiveReal -> PositiveReal
        # Natural x Real -> Real
        # Natural x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(bino, bern).inf_type, Natural)
        self.assertEqual(MultiplicationNode(bino, beta).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(bino, norm).inf_type, Real)
        self.assertEqual(MultiplicationNode(bino, t).inf_type, Tensor)

        # PositiveReal x bool -> PositiveReal
        # PositiveReal x Probability -> PositiveReal
        # PositiveReal x Natural -> PositiveReal
        # PositiveReal x PositiveReal -> PositiveReal
        # PositiveReal x Real -> Real
        # PositiveReal x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(half, bern).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, beta).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, bino).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, half).inf_type, PositiveReal)
        self.assertEqual(MultiplicationNode(half, norm).inf_type, Real)
        self.assertEqual(MultiplicationNode(half, t).inf_type, Tensor)

        # Real x bool -> Real
        # Real x Probability -> Real
        # Real x Natural -> Real
        # Real x PositiveReal -> Real
        # Real x Real -> Real
        # Real x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(norm, bern).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, beta).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, bino).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, half).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, norm).inf_type, Real)
        self.assertEqual(MultiplicationNode(norm, t).inf_type, Tensor)

        # Tensor x bool -> Tensor
        # Tensor x Probability -> Tensor
        # Tensor x Natural -> Tensor
        # Tensor x PositiveReal -> Tensor
        # Tensor x Real -> Tensor
        # Tensor x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(t, bern).inf_type, Tensor)
        self.assertEqual(MultiplicationNode(t, beta).inf_type, Tensor)
        self.assertEqual(MultiplicationNode(t, bino).inf_type, Tensor)
        self.assertEqual(MultiplicationNode(t, half).inf_type, Tensor)
        self.assertEqual(MultiplicationNode(t, norm).inf_type, Tensor)
        self.assertEqual(MultiplicationNode(t, t).inf_type, Tensor)

        # Addition

        # bool + bool -> PositiveReal
        # bool + Probability -> PositiveReal
        # bool + Natural -> PositiveReal
        # bool + PositiveReal -> PositiveReal
        # bool + Real -> Real
        # bool + Tensor -> Tensor
        self.assertEqual(AdditionNode(bern, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, norm).inf_type, Real)
        self.assertEqual(AdditionNode(bern, t).inf_type, Tensor)

        # Probability + bool -> PositiveReal
        # Probability + Probability -> PositiveReal
        # Probability + Natural -> PositiveReal
        # Probability + PositiveReal -> PositiveReal
        # Probability + Real -> Real
        # Probability + Tensor -> Tensor
        self.assertEqual(AdditionNode(beta, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, norm).inf_type, Real)
        self.assertEqual(AdditionNode(beta, t).inf_type, Tensor)

        # Natural + bool -> PositiveReal
        # Natural + Probability -> PositiveReal
        # Natural + Natural -> PositiveReal
        # Natural + PositiveReal -> PositiveReal
        # Natural + Real -> Real
        # Natural + Tensor -> Tensor
        self.assertEqual(AdditionNode(bino, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, norm).inf_type, Real)
        self.assertEqual(AdditionNode(bino, t).inf_type, Tensor)

        # PositiveReal + bool -> PositiveReal
        # PositiveReal + Probability -> PositiveReal
        # PositiveReal + Natural -> PositiveReal
        # PositiveReal + PositiveReal -> PositiveReal
        # PositiveReal + Real -> Real
        # PositiveReal + Tensor -> Tensor
        self.assertEqual(AdditionNode(half, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, norm).inf_type, Real)
        self.assertEqual(AdditionNode(half, t).inf_type, Tensor)

        # Real + bool -> Real
        # Real + Probability -> Real
        # Real + Natural -> Real
        # Real + PositiveReal -> Real
        # Real + Real -> Real
        # Real + Tensor -> Tensor
        self.assertEqual(AdditionNode(norm, bern).inf_type, Real)
        self.assertEqual(AdditionNode(norm, beta).inf_type, Real)
        self.assertEqual(AdditionNode(norm, bino).inf_type, Real)
        self.assertEqual(AdditionNode(norm, half).inf_type, Real)
        self.assertEqual(AdditionNode(norm, norm).inf_type, Real)
        self.assertEqual(AdditionNode(norm, t).inf_type, Tensor)

        # Tensor + bool -> Tensor
        # Tensor + Probability -> Tensor
        # Tensor + Natural -> Tensor
        # Tensor + PositiveReal -> Tensor
        # Tensor + Real -> Tensor
        # Tensor + Tensor -> Tensor
        self.assertEqual(AdditionNode(t, bern).inf_type, Tensor)
        self.assertEqual(AdditionNode(t, beta).inf_type, Tensor)
        self.assertEqual(AdditionNode(t, bino).inf_type, Tensor)
        self.assertEqual(AdditionNode(t, half).inf_type, Tensor)
        self.assertEqual(AdditionNode(t, norm).inf_type, Tensor)
        self.assertEqual(AdditionNode(t, t).inf_type, Tensor)

        # IfThenElse

        # bool : bool -> bool
        # bool : Probability -> Probability
        # bool : Natural -> Natural
        # bool : PositiveReal -> PositiveReal
        # bool : Real -> Real
        # bool : Tensor -> Tensor
        self.assertEqual(IfThenElseNode(bern, bern, bern).inf_type, bool)
        self.assertEqual(IfThenElseNode(bern, bern, beta).inf_type, Probability)
        self.assertEqual(IfThenElseNode(bern, bern, bino).inf_type, Natural)
        self.assertEqual(IfThenElseNode(bern, bern, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, bern, norm).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, bern, t).inf_type, Tensor)

        # Probability : bool -> Probability
        # Probability : Probability -> Probability
        # Probability : Natural -> PositiveReal
        # Probability : PositiveReal -> PositiveReal
        # Probability : Real -> Real
        # Probability : Tensor -> Tensor
        self.assertEqual(IfThenElseNode(bern, beta, bern).inf_type, Probability)
        self.assertEqual(IfThenElseNode(bern, beta, beta).inf_type, Probability)
        self.assertEqual(IfThenElseNode(bern, beta, bino).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, beta, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, beta, norm).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, beta, t).inf_type, Tensor)

        # Natural : bool -> Natural
        # Natural : Probability -> PositiveReal
        # Natural : Natural -> Natural
        # Natural : PositiveReal -> PositiveReal
        # Natural : Real -> Real
        # Natural : Tensor -> Tensor
        self.assertEqual(IfThenElseNode(bern, bino, bern).inf_type, Natural)
        self.assertEqual(IfThenElseNode(bern, bino, beta).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, bino, bino).inf_type, Natural)
        self.assertEqual(IfThenElseNode(bern, bino, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, bino, norm).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, bino, t).inf_type, Tensor)

        # PositiveReal : bool -> PositiveReal
        # PositiveReal : Probability -> PositiveReal
        # PositiveReal : Natural -> PositiveReal
        # PositiveReal : PositiveReal -> PositiveReal
        # PositiveReal : Real -> Real
        # PositiveReal : Tensor -> Tensor
        self.assertEqual(IfThenElseNode(bern, half, bern).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, beta).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, bino).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, half).inf_type, PositiveReal)
        self.assertEqual(IfThenElseNode(bern, half, norm).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, half, t).inf_type, Tensor)

        # Real : bool -> Real
        # Real : Probability -> Real
        # Real : Natural -> Real
        # Real : PositiveReal -> Real
        # Real : Real -> Real
        # Real : Tensor -> Tensor
        self.assertEqual(IfThenElseNode(bern, norm, bern).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, beta).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, bino).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, half).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, norm).inf_type, Real)
        self.assertEqual(IfThenElseNode(bern, norm, t).inf_type, Tensor)

        # Tensor : bool -> Tensor
        # Tensor : Probability -> Tensor
        # Tensor : Natural -> Tensor
        # Tensor : PositiveReal -> Tensor
        # Tensor : Real -> Tensor
        # Tensor : Tensor -> Tensor
        self.assertEqual(IfThenElseNode(bern, t, bern).inf_type, Tensor)
        self.assertEqual(IfThenElseNode(bern, t, beta).inf_type, Tensor)
        self.assertEqual(IfThenElseNode(bern, t, bino).inf_type, Tensor)
        self.assertEqual(IfThenElseNode(bern, t, half).inf_type, Tensor)
        self.assertEqual(IfThenElseNode(bern, t, norm).inf_type, Tensor)
        self.assertEqual(IfThenElseNode(bern, t, t).inf_type, Tensor)

        # Negate
        # - bool -> Real
        # - Probability -> Real
        # - Natural -> Real
        # - PositiveReal -> Real
        # - Real -> Real
        # - Tensor -> Tensor
        self.assertEqual(NegateNode(bern).inf_type, Real)
        self.assertEqual(NegateNode(beta).inf_type, Real)
        self.assertEqual(NegateNode(bino).inf_type, Real)
        self.assertEqual(NegateNode(half).inf_type, Real)
        self.assertEqual(NegateNode(norm).inf_type, Real)
        self.assertEqual(NegateNode(t).inf_type, Tensor)

        # Exp
        # exp bool -> PositiveReal
        # exp Probability -> PositiveReal
        # exp Natural -> PositiveReal
        # exp PositiveReal -> PositiveReal
        # exp Real -> PositiveReal
        # exp Tensor -> Tensor
        self.assertEqual(ExpNode(bern).inf_type, PositiveReal)
        self.assertEqual(ExpNode(beta).inf_type, PositiveReal)
        self.assertEqual(ExpNode(bino).inf_type, PositiveReal)
        self.assertEqual(ExpNode(half).inf_type, PositiveReal)
        self.assertEqual(ExpNode(norm).inf_type, PositiveReal)
        self.assertEqual(ExpNode(t).inf_type, Tensor)

        # To Real
        self.assertEqual(ToRealNode(bern).inf_type, Real)
        self.assertEqual(ToRealNode(beta).inf_type, Real)
        self.assertEqual(ToRealNode(bino).inf_type, Real)
        self.assertEqual(ToRealNode(half).inf_type, Real)
        self.assertEqual(ToRealNode(norm).inf_type, Real)
        # To Real is illegal on tensors

        # To Tensor
        self.assertEqual(ToTensorNode(bern).inf_type, Tensor)
        self.assertEqual(ToTensorNode(beta).inf_type, Tensor)
        self.assertEqual(ToTensorNode(bino).inf_type, Tensor)
        self.assertEqual(ToTensorNode(half).inf_type, Tensor)
        self.assertEqual(ToTensorNode(norm).inf_type, Tensor)
        self.assertEqual(ToTensorNode(t).inf_type, Tensor)

        # To Positive Real
        self.assertEqual(ToPositiveRealNode(bern).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(beta).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(bino).inf_type, PositiveReal)
        self.assertEqual(ToPositiveRealNode(half).inf_type, PositiveReal)
        # To Positive Real is illegal on reals and tensors

        # Power
        # * The inf type is equal to the base inf type with a few exceptions:
        # * If the base is P or B and the exponent is R, the output is R+.
        # * If the base is B the output is P.
        # * If the base is N the output is R+.
        # TODO: We can do a slightly better job than this; see comments
        # in bmg_nodes.py for details.

        # Base is B
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

        # Base is N
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
        t = TensorNode(tensor([1.0, 1.0]))

        self.assertEqual(b.requirements, [])
        self.assertEqual(prob.requirements, [])
        self.assertEqual(pos.requirements, [])
        self.assertEqual(real.requirements, [])
        self.assertEqual(nat.requirements, [])
        self.assertEqual(t.requirements, [])

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

        bern = SampleNode(BernoulliNode(prob))
        beta = SampleNode(BetaNode(pos, pos))
        bino = SampleNode(BinomialNode(nat, prob))
        half = SampleNode(HalfCauchyNode(pos))
        norm = SampleNode(NormalNode(real, pos))

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
        #
        # The question then is: what requirements should be generated on the edges
        # when we have, say, a bool multiplied by a probability? The smallest possible
        # requirement is just that: the bool is required to be a bool, and the
        # probability is required to be a probability, and we can "multiply" them
        # by converting the multiplication to an if-then-else.
        #
        # Consider by contrast a multiplication of probability by natural.
        # There is no multiplication node or equivalent node we can generate that
        # is any better than "convert both to positive real and multiply them",
        # so in that case we put a requirement on both operands that they be
        # positive reals.

        # bool x bool -> bool
        # bool x Probability -> Probability
        # bool x Natural -> Natural
        # bool x PositiveReal -> PositiveReal
        # bool x Real -> Real
        # bool x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(bern, bern).requirements, [bool, bool])
        self.assertEqual(
            MultiplicationNode(bern, beta).requirements, [bool, Probability]
        )
        self.assertEqual(MultiplicationNode(bern, bino).requirements, [bool, Natural])
        self.assertEqual(
            MultiplicationNode(bern, half).requirements, [bool, PositiveReal]
        )
        self.assertEqual(MultiplicationNode(bern, norm).requirements, [bool, Real])
        self.assertEqual(MultiplicationNode(bern, t).requirements, [bool, Tensor])

        # Probability x bool -> Probability
        # Probability x Probability -> Probability
        # Probability x Natural -> PositiveReal
        # Probability x PositiveReal -> PositiveReal
        # Probability x Real -> Real
        # Probability x Tensor -> Tensor
        self.assertEqual(
            MultiplicationNode(beta, bern).requirements, [Probability, bool]
        )
        self.assertEqual(
            MultiplicationNode(beta, beta).requirements, [Probability, Probability]
        )
        self.assertEqual(
            MultiplicationNode(beta, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            MultiplicationNode(beta, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(MultiplicationNode(beta, norm).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(beta, t).requirements, [Tensor, Tensor])

        # Natural x bool -> Natural
        # Natural x Probability -> PositiveReal
        # Natural x Natural -> PositiveReal
        # Natural x PositiveReal -> PositiveReal
        # Natural x Real -> Real
        # Natural x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(bino, bern).requirements, [Natural, bool])
        self.assertEqual(
            MultiplicationNode(bino, beta).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            MultiplicationNode(bino, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            MultiplicationNode(bino, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(MultiplicationNode(bino, norm).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(bino, t).requirements, [Tensor, Tensor])

        # PositiveReal x bool -> PositiveReal
        # PositiveReal x Probability -> PositiveReal
        # PositiveReal x Natural -> PositiveReal
        # PositiveReal x PositiveReal -> PositiveReal
        # PositiveReal x Real -> Real
        # PositiveReal x Tensor -> Tensor
        self.assertEqual(
            MultiplicationNode(half, bern).requirements, [PositiveReal, bool]
        )
        self.assertEqual(
            MultiplicationNode(half, beta).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            MultiplicationNode(half, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            MultiplicationNode(half, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(MultiplicationNode(half, norm).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(half, t).requirements, [Tensor, Tensor])

        # Real x bool -> Real
        # Real x Probability -> Real
        # Real x Natural -> Real
        # Real x PositiveReal -> Real
        # Real x Real -> Real
        # Real x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(norm, bern).requirements, [Real, bool])
        self.assertEqual(MultiplicationNode(norm, beta).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, bino).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, half).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, norm).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, t).requirements, [Tensor, Tensor])

        # Tensor x bool -> Tensor
        # Tensor x Probability -> Tensor
        # Tensor x Natural -> Tensor
        # Tensor x PositiveReal -> Tensor
        # Tensor x Real -> Tensor
        # Tensor x Tensor -> Tensor
        self.assertEqual(MultiplicationNode(t, bern).requirements, [Tensor, bool])
        self.assertEqual(MultiplicationNode(t, beta).requirements, [Tensor, Tensor])
        self.assertEqual(MultiplicationNode(t, bino).requirements, [Tensor, Tensor])
        self.assertEqual(MultiplicationNode(t, half).requirements, [Tensor, Tensor])
        self.assertEqual(MultiplicationNode(t, norm).requirements, [Tensor, Tensor])
        self.assertEqual(MultiplicationNode(t, t).requirements, [Tensor, Tensor])

        # Addition

        # bool + bool -> PositiveReal
        # bool + Probability -> PositiveReal
        # bool + Natural -> PositiveReal
        # bool + PositiveReal -> PositiveReal
        # bool + Real -> Real
        # bool + Tensor -> Tensor
        self.assertEqual(
            AdditionNode(bern, bern).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(bern, beta).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(bern, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(bern, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(AdditionNode(bern, norm).requirements, [Real, Real])
        self.assertEqual(AdditionNode(bern, t).requirements, [Tensor, Tensor])

        # Probability + bool -> PositiveReal
        # Probability + Probability -> PositiveReal
        # Probability + Natural -> PositiveReal
        # Probability + PositiveReal -> PositiveReal
        # Probability + Real -> Real
        # Probability + Tensor -> Tensor
        self.assertEqual(
            AdditionNode(beta, bern).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(beta, beta).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(beta, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(beta, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(AdditionNode(beta, norm).requirements, [Real, Real])
        self.assertEqual(AdditionNode(beta, t).requirements, [Tensor, Tensor])

        # Natural + bool -> PositiveReal
        # Natural + Probability -> PositiveReal
        # Natural + Natural -> PositiveReal
        # Natural + PositiveReal -> PositiveReal
        # Natural + Real -> Real
        # Natural + Tensor -> Tensor
        self.assertEqual(
            AdditionNode(bino, bern).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(bino, beta).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(bino, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(bino, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(AdditionNode(bino, norm).requirements, [Real, Real])
        self.assertEqual(AdditionNode(bino, t).requirements, [Tensor, Tensor])

        # PositiveReal + bool -> PositiveReal
        # PositiveReal + Probability -> PositiveReal
        # PositiveReal + Natural -> PositiveReal
        # PositiveReal + PositiveReal -> PositiveReal
        # PositiveReal + Real -> Real
        # PositiveReal + Tensor -> Tensor
        self.assertEqual(
            AdditionNode(half, bern).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(half, beta).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(half, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            AdditionNode(half, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(AdditionNode(half, norm).requirements, [Real, Real])
        self.assertEqual(AdditionNode(half, t).requirements, [Tensor, Tensor])

        # Real + bool -> Real
        # Real + Probability -> Real
        # Real + Natural -> Real
        # Real + PositiveReal -> Real
        # Real + Real -> Real
        # Real + Tensor -> Tensor
        self.assertEqual(AdditionNode(norm, bern).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, beta).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, bino).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, half).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, norm).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, t).requirements, [Tensor, Tensor])

        # Tensor + bool -> Tensor
        # Tensor + Probability -> Tensor
        # Tensor + Natural -> Tensor
        # Tensor + PositiveReal -> Tensor
        # Tensor + Real -> Tensor
        # Tensor + Tensor -> Tensor
        self.assertEqual(AdditionNode(t, bern).requirements, [Tensor, Tensor])
        self.assertEqual(AdditionNode(t, beta).requirements, [Tensor, Tensor])
        self.assertEqual(AdditionNode(t, bino).requirements, [Tensor, Tensor])
        self.assertEqual(AdditionNode(t, half).requirements, [Tensor, Tensor])
        self.assertEqual(AdditionNode(t, norm).requirements, [Tensor, Tensor])
        self.assertEqual(AdditionNode(t, t).requirements, [Tensor, Tensor])

        # IfThenElse

        # bool : bool -> bool
        # bool : Probability -> Probability
        # bool : Natural -> Natural
        # bool : PositiveReal -> PositiveReal
        # bool : Real -> Real
        # bool : Tensor -> Tensor
        self.assertEqual(
            IfThenElseNode(bern, bern, bern).requirements, [bool, bool, bool]
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, beta).requirements,
            [bool, Probability, Probability],
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, bino).requirements, [bool, Natural, Natural]
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, half).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, norm).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, t).requirements, [bool, Tensor, Tensor]
        )

        # Probability : bool -> Probability
        # Probability : Probability -> Probability
        # Probability : Natural -> PositiveReal
        # Probability : PositiveReal -> PositiveReal
        # Probability : Real -> Real
        # Probability : Tensor -> Tensor
        self.assertEqual(
            IfThenElseNode(bern, beta, bern).requirements,
            [bool, Probability, Probability],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, beta).requirements,
            [bool, Probability, Probability],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, bino).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, half).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, norm).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, t).requirements, [bool, Tensor, Tensor]
        )

        # Natural : bool -> Natural
        # Natural : Probability -> PositiveReal
        # Natural : Natural -> Natural
        # Natural : PositiveReal -> PositiveReal
        # Natural : Real -> Real
        # Natural : Tensor -> Tensor
        self.assertEqual(
            IfThenElseNode(bern, bino, bern).requirements, [bool, Natural, Natural]
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, beta).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, bino).requirements, [bool, Natural, Natural]
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, half).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, norm).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, t).requirements, [bool, Tensor, Tensor]
        )

        # PositiveReal : bool -> PositiveReal
        # PositiveReal : Probability -> PositiveReal
        # PositiveReal : Natural -> PositiveReal
        # PositiveReal : PositiveReal -> PositiveReal
        # PositiveReal : Real -> Real
        # PositiveReal : Tensor -> Tensor
        self.assertEqual(
            IfThenElseNode(bern, half, bern).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, beta).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, bino).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, half).requirements,
            [bool, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, norm).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, half, t).requirements, [bool, Tensor, Tensor]
        )

        # Real : bool -> Real
        # Real : Probability -> Real
        # Real : Natural -> Real
        # Real : PositiveReal -> Real
        # Real : Real -> Real
        # Real : Tensor -> Tensor
        self.assertEqual(
            IfThenElseNode(bern, norm, bern).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, beta).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, bino).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, half).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, norm).requirements, [bool, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, t).requirements, [bool, Tensor, Tensor]
        )

        # Tensor : bool -> Tensor
        # Tensor : Probability -> Tensor
        # Tensor : Natural -> Tensor
        # Tensor : PositiveReal -> Tensor
        # Tensor : Real -> Tensor
        # Tensor : Tensor -> Tensor
        self.assertEqual(
            IfThenElseNode(bern, t, bern).requirements, [bool, Tensor, Tensor]
        )
        self.assertEqual(
            IfThenElseNode(bern, t, beta).requirements, [bool, Tensor, Tensor]
        )
        self.assertEqual(
            IfThenElseNode(bern, t, bino).requirements, [bool, Tensor, Tensor]
        )
        self.assertEqual(
            IfThenElseNode(bern, t, half).requirements, [bool, Tensor, Tensor]
        )
        self.assertEqual(
            IfThenElseNode(bern, t, norm).requirements, [bool, Tensor, Tensor]
        )
        self.assertEqual(
            IfThenElseNode(bern, t, t).requirements, [bool, Tensor, Tensor]
        )

        # Negate
        # - bool -> Real
        # - Probability -> Real
        # - Natural -> Real
        # - PositiveReal -> Real
        # - Real -> Real
        # - Tensor -> Tensor
        self.assertEqual(NegateNode(bern).requirements, [Real])
        self.assertEqual(NegateNode(beta).requirements, [Real])
        self.assertEqual(NegateNode(bino).requirements, [Real])
        self.assertEqual(NegateNode(half).requirements, [Real])
        self.assertEqual(NegateNode(norm).requirements, [Real])
        self.assertEqual(NegateNode(t).requirements, [Tensor])

        # Exp
        # Exp requires that its operand be positive real, real
        # or tensor.

        self.assertEqual(ExpNode(bern).requirements, [PositiveReal])
        self.assertEqual(ExpNode(beta).requirements, [PositiveReal])
        self.assertEqual(ExpNode(bino).requirements, [PositiveReal])
        self.assertEqual(ExpNode(half).requirements, [PositiveReal])
        self.assertEqual(ExpNode(norm).requirements, [Real])
        self.assertEqual(ExpNode(t).requirements, [Tensor])

        # To Real
        self.assertEqual(ToRealNode(bern).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(beta).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(bino).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(half).requirements, [upper_bound(Real)])
        self.assertEqual(ToRealNode(norm).requirements, [upper_bound(Real)])
        # To Real is illegal on tensors

        # To Tensor
        self.assertEqual(ToTensorNode(bern).requirements, [upper_bound(Tensor)])
        self.assertEqual(ToTensorNode(beta).requirements, [upper_bound(Tensor)])
        self.assertEqual(ToTensorNode(bino).requirements, [upper_bound(Tensor)])
        self.assertEqual(ToTensorNode(half).requirements, [upper_bound(Tensor)])
        self.assertEqual(ToTensorNode(norm).requirements, [upper_bound(Tensor)])
        self.assertEqual(ToTensorNode(t).requirements, [upper_bound(Tensor)])

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

        # Power
        #
        # For non-tensor cases we require that the base be P, R+, R and the
        # exponent be R+ or R.
        #
        # TODO: We can do a slightly better job than this; see comments
        # in bmg_nodes.py for details.

        self.assertEqual(
            PowerNode(bern, bern).requirements, [Probability, PositiveReal]
        )
        self.assertEqual(
            PowerNode(beta, beta).requirements, [Probability, PositiveReal]
        )
        self.assertEqual(
            PowerNode(bino, bino).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(
            PowerNode(half, half).requirements, [PositiveReal, PositiveReal]
        )
        self.assertEqual(PowerNode(norm, norm).requirements, [Real, Real])
