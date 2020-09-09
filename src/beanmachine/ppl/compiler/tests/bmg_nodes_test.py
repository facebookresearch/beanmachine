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
    ComplementNode,
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
    ToPositiveRealNode,
    ToRealNode,
)
from beanmachine.ppl.compiler.bmg_types import (
    Boolean,
    Natural,
    One,
    PositiveReal,
    Probability,
    Real,
    upper_bound,
)


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
        bf = BooleanNode(False)
        prob = ProbabilityNode(0.5)
        pos = PositiveRealNode(1.5)
        real = RealNode(-1.5)
        nat = NaturalNode(2)

        self.assertEqual(b.inf_type, One)
        self.assertEqual(bf.inf_type, Boolean)
        self.assertEqual(prob.inf_type, Probability)
        self.assertEqual(pos.inf_type, PositiveReal)
        self.assertEqual(real.inf_type, Real)
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

        # Boolean x Boolean -> Boolean
        # Boolean x Probability -> Probability
        # Boolean x Natural -> Natural
        # Boolean x PositiveReal -> PositiveReal
        # Boolean x Real -> Real
        self.assertEqual(MultiplicationNode(bern, bern).inf_type, Boolean)
        self.assertEqual(MultiplicationNode(bern, beta).inf_type, Probability)
        self.assertEqual(MultiplicationNode(bern, bino).inf_type, Natural)
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

        # Natural x Boolean -> Natural
        # Natural x Probability -> PositiveReal
        # Natural x Natural -> PositiveReal
        # Natural x PositiveReal -> PositiveReal
        # Natural x Real -> Real
        self.assertEqual(MultiplicationNode(bino, bern).inf_type, Natural)
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

        # Special case: 1 + (-P) -> P
        # Special case: 1 + (-B) -> B
        # Special case: (-P) + 1 -> P
        # Special case: (-B) + 1 -> B
        nb = NegateNode(bern)
        np = NegateNode(beta)
        one = RealNode(1.0)
        self.assertEqual(AdditionNode(np, one).inf_type, Probability)
        self.assertEqual(AdditionNode(nb, one).inf_type, Boolean)
        self.assertEqual(AdditionNode(one, np).inf_type, Probability)
        self.assertEqual(AdditionNode(one, nb).inf_type, Boolean)

        # Boolean + Boolean -> PositiveReal
        # Boolean + Probability -> PositiveReal
        # Boolean + Natural -> PositiveReal
        # Boolean + PositiveReal -> PositiveReal
        # Boolean + Real -> Real
        self.assertEqual(AdditionNode(bern, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bern, norm).inf_type, Real)

        # Probability + Boolean -> PositiveReal
        # Probability + Probability -> PositiveReal
        # Probability + Natural -> PositiveReal
        # Probability + PositiveReal -> PositiveReal
        # Probability + Real -> Real
        self.assertEqual(AdditionNode(beta, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(beta, norm).inf_type, Real)

        # Natural + Boolean -> PositiveReal
        # Natural + Probability -> PositiveReal
        # Natural + Natural -> PositiveReal
        # Natural + PositiveReal -> PositiveReal
        # Natural + Real -> Real
        self.assertEqual(AdditionNode(bino, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(bino, norm).inf_type, Real)

        # PositiveReal + Boolean -> PositiveReal
        # PositiveReal + Probability -> PositiveReal
        # PositiveReal + Natural -> PositiveReal
        # PositiveReal + PositiveReal -> PositiveReal
        # PositiveReal + Real -> Real
        self.assertEqual(AdditionNode(half, bern).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, beta).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, bino).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, half).inf_type, PositiveReal)
        self.assertEqual(AdditionNode(half, norm).inf_type, Real)

        # Real + Boolean -> Real
        # Real + Probability -> Real
        # Real + Natural -> Real
        # Real + PositiveReal -> Real
        # Real + Real -> Real
        self.assertEqual(AdditionNode(norm, bern).inf_type, Real)
        self.assertEqual(AdditionNode(norm, beta).inf_type, Real)
        self.assertEqual(AdditionNode(norm, bino).inf_type, Real)
        self.assertEqual(AdditionNode(norm, half).inf_type, Real)
        self.assertEqual(AdditionNode(norm, norm).inf_type, Real)

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

        # Power
        # * The inf type is equal to the base inf type with a few exceptions:
        # * If the base is P or B and the exponent is R, the output is R+.
        # * If the base is B the output is P.
        # * If the base is N the output is R+.

        # Base is B
        self.assertEqual(PowerNode(bern, bern).inf_type, Boolean)
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
        self.assertEqual(PowerNode(bino, bern).inf_type, Natural)
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

        # Boolean x Boolean -> Boolean
        # Boolean x Probability -> Probability
        # Boolean x Natural -> Natural
        # Boolean x PositiveReal -> PositiveReal
        # Boolean x Real -> Real
        self.assertEqual(
            MultiplicationNode(bern, bern).requirements, [Boolean, Boolean]
        )
        self.assertEqual(
            MultiplicationNode(bern, beta).requirements, [Boolean, Probability]
        )
        self.assertEqual(
            MultiplicationNode(bern, bino).requirements, [Boolean, Natural]
        )
        self.assertEqual(
            MultiplicationNode(bern, half).requirements, [Boolean, PositiveReal]
        )
        self.assertEqual(MultiplicationNode(bern, norm).requirements, [Boolean, Real])

        # Probability x Boolean -> Probability
        # Probability x Probability -> Probability
        # Probability x Natural -> PositiveReal
        # Probability x PositiveReal -> PositiveReal
        # Probability x Real -> Real
        self.assertEqual(
            MultiplicationNode(beta, bern).requirements, [Probability, Boolean]
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

        # Natural x Boolean -> Natural
        # Natural x Probability -> PositiveReal
        # Natural x Natural -> PositiveReal
        # Natural x PositiveReal -> PositiveReal
        # Natural x Real -> Real
        self.assertEqual(
            MultiplicationNode(bino, bern).requirements, [Natural, Boolean]
        )
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

        # PositiveReal x Boolean -> PositiveReal
        # PositiveReal x Probability -> PositiveReal
        # PositiveReal x Natural -> PositiveReal
        # PositiveReal x PositiveReal -> PositiveReal
        # PositiveReal x Real -> Real
        self.assertEqual(
            MultiplicationNode(half, bern).requirements, [PositiveReal, Boolean]
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

        # Real x Boolean -> Real
        # Real x Probability -> Real
        # Real x Natural -> Real
        # Real x PositiveReal -> Real
        # Real x Real -> Real
        self.assertEqual(MultiplicationNode(norm, bern).requirements, [Real, Boolean])
        self.assertEqual(MultiplicationNode(norm, beta).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, bino).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, half).requirements, [Real, Real])
        self.assertEqual(MultiplicationNode(norm, norm).requirements, [Real, Real])

        # Addition

        # Boolean + Boolean -> PositiveReal
        # Boolean + Probability -> PositiveReal
        # Boolean + Natural -> PositiveReal
        # Boolean + PositiveReal -> PositiveReal
        # Boolean + Real -> Real
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

        # Probability + Boolean -> PositiveReal
        # Probability + Probability -> PositiveReal
        # Probability + Natural -> PositiveReal
        # Probability + PositiveReal -> PositiveReal
        # Probability + Real -> Real
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

        # Natural + Boolean -> PositiveReal
        # Natural + Probability -> PositiveReal
        # Natural + Natural -> PositiveReal
        # Natural + PositiveReal -> PositiveReal
        # Natural + Real -> Real
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

        # PositiveReal + Boolean -> PositiveReal
        # PositiveReal + Probability -> PositiveReal
        # PositiveReal + Natural -> PositiveReal
        # PositiveReal + PositiveReal -> PositiveReal
        # PositiveReal + Real -> Real
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

        # Real + Boolean -> Real
        # Real + Probability -> Real
        # Real + Natural -> Real
        # Real + PositiveReal -> Real
        # Real + Real -> Real
        self.assertEqual(AdditionNode(norm, bern).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, beta).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, bino).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, half).requirements, [Real, Real])
        self.assertEqual(AdditionNode(norm, norm).requirements, [Real, Real])

        # IfThenElse

        # Boolean : Boolean -> Boolean
        # Boolean : Probability -> Probability
        # Boolean : Natural -> Natural
        # Boolean : PositiveReal -> PositiveReal
        # Boolean : Real -> Real
        self.assertEqual(
            IfThenElseNode(bern, bern, bern).requirements, [Boolean, Boolean, Boolean]
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, beta).requirements,
            [Boolean, Probability, Probability],
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, bino).requirements, [Boolean, Natural, Natural]
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, half).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, bern, norm).requirements, [Boolean, Real, Real]
        )

        # Probability : Boolean -> Probability
        # Probability : Probability -> Probability
        # Probability : Natural -> PositiveReal
        # Probability : PositiveReal -> PositiveReal
        # Probability : Real -> Real
        self.assertEqual(
            IfThenElseNode(bern, beta, bern).requirements,
            [Boolean, Probability, Probability],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, beta).requirements,
            [Boolean, Probability, Probability],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, bino).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, half).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, beta, norm).requirements, [Boolean, Real, Real]
        )

        # Natural : Boolean -> Natural
        # Natural : Probability -> PositiveReal
        # Natural : Natural -> Natural
        # Natural : PositiveReal -> PositiveReal
        # Natural : Real -> Real
        self.assertEqual(
            IfThenElseNode(bern, bino, bern).requirements, [Boolean, Natural, Natural]
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, beta).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, bino).requirements, [Boolean, Natural, Natural]
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, half).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, bino, norm).requirements, [Boolean, Real, Real]
        )

        # PositiveReal : Boolean -> PositiveReal
        # PositiveReal : Probability -> PositiveReal
        # PositiveReal : Natural -> PositiveReal
        # PositiveReal : PositiveReal -> PositiveReal
        # PositiveReal : Real -> Real
        self.assertEqual(
            IfThenElseNode(bern, half, bern).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, beta).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, bino).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, half).requirements,
            [Boolean, PositiveReal, PositiveReal],
        )
        self.assertEqual(
            IfThenElseNode(bern, half, norm).requirements, [Boolean, Real, Real]
        )

        # Real : Boolean -> Real
        # Real : Probability -> Real
        # Real : Natural -> Real
        # Real : PositiveReal -> Real
        # Real : Real -> Real
        self.assertEqual(
            IfThenElseNode(bern, norm, bern).requirements, [Boolean, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, beta).requirements, [Boolean, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, bino).requirements, [Boolean, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, half).requirements, [Boolean, Real, Real]
        )
        self.assertEqual(
            IfThenElseNode(bern, norm, norm).requirements, [Boolean, Real, Real]
        )

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

        # Power
        #
        # We require that the base be P, R+, R and the exponent be R+ or R.
        # However, if the exponent is bool then the base can be any type
        # because we can rewrite the whole thing into an if-then-else.

        self.assertEqual(PowerNode(bino, bern).requirements, [Natural, Boolean])
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
