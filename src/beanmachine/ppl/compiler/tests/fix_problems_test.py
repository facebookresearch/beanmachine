# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for fix_problems.py"""
import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.fix_problems import fix_problems
from beanmachine.ppl.compiler.gen_dot import to_dot
from torch import tensor


class FixProblemsTest(unittest.TestCase):
    def test_fix_problems_01(self) -> None:
        # Problems that need to be fixed:
        #
        # * Single-valued tensors are used in contexts where scalars are needed.
        # * A multiplication of 0.5 by a probability (from a beta) is used both
        #    as a probability (by a Bernoulli) and a real (by a normal).
        #
        # The solutions:
        #
        # * The constants are replaced by constants of the appropriate kinds.
        # * A to-real node is inserted between the multiplication and the normal.
        #

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
            inf_types=False,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N00[label="0.5:T"];
  N01[label="2.0:T"];
  N02[label="Beta:P"];
  N03[label="Sample:P"];
  N04[label="*:M"];
  N05[label="1.0:T"];
  N06[label="Normal:R"];
  N07[label="Sample:R"];
  N08[label="Bernoulli:B"];
  N09[label="Sample:B"];
  N10[label="Query:M"];
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

        fix_problems(bmg)
        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=False,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N00[label="1.0:T"];
  N01[label="2.0:T"];
  N02[label="0.5:T"];
  N03[label="0.5:P"];
  N04[label="2.0:R+"];
  N05[label="Beta:P"];
  N06[label="Sample:P"];
  N07[label="*:P"];
  N08[label="ToReal:R"];
  N09[label="1.0:R+"];
  N10[label="Normal:R"];
  N11[label="Sample:R"];
  N12[label="Bernoulli:B"];
  N13[label="Sample:B"];
  N14[label="Query:P"];
  N03 -> N07[label="left:P"];
  N04 -> N05[label="alpha:R+"];
  N04 -> N05[label="beta:R+"];
  N05 -> N06[label="operand:P"];
  N06 -> N07[label="right:P"];
  N07 -> N08[label="operand:<=R"];
  N07 -> N12[label="probability:P"];
  N07 -> N14[label="operator:any"];
  N08 -> N10[label="mu:R"];
  N09 -> N10[label="sigma:R+"];
  N10 -> N11[label="operand:R"];
  N12 -> N13[label="operand:B"];
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_2(self) -> None:
        """test_fix_problems_2"""

        # Problems that need to be fixed:
        #
        # * Single-valued tensors are used in contexts where scalars are needed.
        # * A Boolean (from a Bernoulli) is used in an addition to make a positive real.
        # * The Boolean is also used as a real and a natural.
        #
        # The solutions:
        #
        # * The constants are replaced by constants of the appropriate kinds.
        # * A to-positive-real node is inserted between the addition and the Bernoulli.
        # * A to-real node is inserted between the normal and the Bernoulli
        # * An if-then-else is inserted to make the Bernoulli into a natural.
        #

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def bern():
        #   return Bernoulli(tensor(0.5))
        # @rv def norm():
        #   return Normal(bern(), bern() + tensor(1.0))
        # @rv def bino():
        #   return Binomial(bern(), 0.5)

        one = bmg.add_constant(tensor(1.0))
        half = bmg.add_constant(tensor(0.5))
        bern = bmg.add_bernoulli(half)
        berns = bmg.add_sample(bern)
        plus = bmg.add_addition(berns, one)
        norm = bmg.add_normal(berns, plus)
        bino = bmg.add_binomial(berns, half)
        bmg.add_sample(norm)
        bmg.add_sample(bino)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=False,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N0[label="0.5:T"];
  N1[label="Bernoulli:B"];
  N2[label="Sample:B"];
  N3[label="1.0:T"];
  N4[label="+:M"];
  N5[label="Normal:R"];
  N6[label="Sample:R"];
  N7[label="Binomial:N"];
  N8[label="Sample:N"];
  N0 -> N1[label="probability:P"];
  N0 -> N7[label="probability:P"];
  N1 -> N2[label="operand:B"];
  N2 -> N4[label="left:R+"];
  N2 -> N5[label="mu:R"];
  N2 -> N7[label="count:N"];
  N3 -> N4[label="right:R+"];
  N4 -> N5[label="sigma:R+"];
  N5 -> N6[label="operand:R"];
  N7 -> N8[label="operand:N"];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        fix_problems(bmg)
        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=False,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N00[label="1.0:T"];
  N01[label="0.5:T"];
  N02[label="0.5:P"];
  N03[label="Bernoulli:B"];
  N04[label="Sample:B"];
  N05[label="ToReal:R"];
  N06[label="ToPosReal:R+"];
  N07[label="1.0:R+"];
  N08[label="+:R+"];
  N09[label="Normal:R"];
  N10[label="Sample:R"];
  N11[label="1:N"];
  N12[label="0:N"];
  N13[label="if:N"];
  N14[label="Binomial:N"];
  N15[label="Sample:N"];
  N02 -> N03[label="probability:P"];
  N02 -> N14[label="probability:P"];
  N03 -> N04[label="operand:B"];
  N04 -> N05[label="operand:<=R"];
  N04 -> N06[label="operand:<=R+"];
  N04 -> N13[label="condition:B"];
  N05 -> N09[label="mu:R"];
  N06 -> N08[label="left:R+"];
  N07 -> N08[label="right:R+"];
  N08 -> N09[label="sigma:R+"];
  N09 -> N10[label="operand:R"];
  N11 -> N13[label="consequence:B"];
  N12 -> N13[label="alternative:B"];
  N13 -> N14[label="count:N"];
  N14 -> N15[label="operand:N"];
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_3(self) -> None:
        """test_fix_problems_3"""

        # This test has some problems that cannot be fixed.
        #
        # * Two-valued tensor constant used as probability
        # * Negative number for standard deviation
        # * Fraction used for count
        # * Number greater than 1.0 used as probability

        # @rv def bern():
        #   return Bernoulli(tensor([0.5, 0.5]))
        # @rv def norm():
        #   return Normal(-1.0, -1.0)
        # @rv def bino():
        #   return Binomial(3.14, 3.14)

        self.maxDiff = None
        bmg = BMGraphBuilder()

        pi = bmg.add_constant(3.14)
        mone = bmg.add_constant(-1.0)
        half = bmg.add_constant(tensor([0.5, 0.5]))
        bern = bmg.add_bernoulli(half)
        norm = bmg.add_normal(mone, mone)
        bino = bmg.add_binomial(pi, pi)
        bmg.add_sample(bern)
        bmg.add_sample(norm)
        bmg.add_sample(bino)

        error_report = fix_problems(bmg)
        observed = str(error_report)
        expected = """
The count of a Binomial is required to be a natural but is a positive real.
The probability of a Bernoulli is required to be a probability but is a 1 x 2 simplex matrix.
The probability of a Binomial is required to be a probability but is a positive real.
The sigma of a Normal is required to be a positive real but is a negative real.
        """
        self.assertEqual(observed.strip(), expected.strip())

    def test_fix_problems_4(self) -> None:
        """test_fix_problems_4"""

        # The problem we have here is:
        #
        # * Multiplication is only defined on probability or larger
        # * We have a multiplication of a bool by a natural
        # * We require a natural.
        #
        # In this scenario, the problem fixer turns the multplication
        # into an if-then-else.
        #
        # @rv def berns():
        #   return Bernoulli(0.5)
        # @rv def nats():
        #   return Binomial(2, 0.5)
        # @rv def bino():
        #   return Binomial(berns() * nats(), 0.5)

        self.maxDiff = None
        bmg = BMGraphBuilder()

        two = bmg.add_natural(2)
        half = bmg.add_probability(0.5)
        bern = bmg.add_bernoulli(half)
        berns = bmg.add_sample(bern)
        nat = bmg.add_binomial(two, half)
        nats = bmg.add_sample(nat)
        mult = bmg.add_multiplication(berns, nats)
        bino = bmg.add_binomial(mult, half)
        bmg.add_sample(bino)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="0.5:P>=P"];
  N1[label="Bernoulli:B>=B"];
  N2[label="Sample:B>=B"];
  N3[label="2:N>=N"];
  N4[label="Binomial:N>=N"];
  N5[label="Sample:N>=N"];
  N6[label="*:M>=R+"];
  N7[label="Binomial:N>=N"];
  N8[label="Sample:N>=N"];
  N0 -> N1[label="probability:P"];
  N0 -> N4[label="probability:P"];
  N0 -> N7[label="probability:P"];
  N1 -> N2[label="operand:B"];
  N2 -> N6[label="left:B"];
  N3 -> N4[label="count:N"];
  N4 -> N5[label="operand:N"];
  N5 -> N6[label="right:N"];
  N6 -> N7[label="count:N"];
  N7 -> N8[label="operand:N"];
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N00[label="0.5:P>=P"];
  N01[label="Bernoulli:B>=B"];
  N02[label="Sample:B>=B"];
  N03[label="2:N>=N"];
  N04[label="Binomial:N>=N"];
  N05[label="Sample:N>=N"];
  N06[label="*:M>=R+"];
  N07[label="0:N>=Z"];
  N08[label="if:N>=N"];
  N09[label="Binomial:N>=N"];
  N10[label="Sample:N>=N"];
  N11[label="0.0:R>=Z"];
  N00 -> N01[label="probability:P"];
  N00 -> N04[label="probability:P"];
  N00 -> N09[label="probability:P"];
  N01 -> N02[label="operand:B"];
  N02 -> N06[label="left:B"];
  N02 -> N08[label="condition:B"];
  N03 -> N04[label="count:N"];
  N04 -> N05[label="operand:N"];
  N05 -> N06[label="right:N"];
  N05 -> N08[label="consequence:N"];
  N07 -> N08[label="alternative:N"];
  N08 -> N09[label="count:N"];
  N09 -> N10[label="operand:N"];
}"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_5(self) -> None:
        """test_fix_problems_5"""

        # Division becomes power.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def hcs(n):
        #   return HalfCauchy(1.0)
        # @rv def norm():
        #   return Normal(log(hcs(3) ** (hcs(1) / hcs(2))), 1.0)

        one = bmg.add_constant(1.0)
        hc = bmg.add_halfcauchy(one)
        hcs1 = bmg.add_sample(hc)
        hcs2 = bmg.add_sample(hc)
        hcs3 = bmg.add_sample(hc)
        q = bmg.add_division(hcs1, hcs2)
        p = bmg.add_power(hcs3, q)
        lg = bmg.add_log(p)
        norm = bmg.add_normal(lg, one)
        bmg.add_sample(norm)

        error_report = fix_problems(bmg)
        observed = str(error_report)
        expected = ""
        self.assertEqual(observed.strip(), expected.strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N00[label="1.0:R>=OH"];
  N01[label="1.0:R+>=OH"];
  N02[label="HalfCauchy:R+>=R+"];
  N03[label="Sample:R+>=R+"];
  N04[label="Sample:R+>=R+"];
  N05[label="/:R+>=R+"];
  N06[label="Sample:R+>=R+"];
  N07[label="-1.0:R>=R-"];
  N08[label="**:R+>=R+"];
  N09[label="*:R+>=R+"];
  N10[label="**:R+>=R+"];
  N11[label="Log:R>=R"];
  N12[label="Normal:R>=R"];
  N13[label="Sample:R>=R"];
  N01 -> N02[label="scale:R+"];
  N01 -> N12[label="sigma:R+"];
  N02 -> N03[label="operand:R+"];
  N02 -> N04[label="operand:R+"];
  N02 -> N06[label="operand:R+"];
  N03 -> N05[label="left:R+"];
  N03 -> N09[label="left:R+"];
  N04 -> N05[label="right:R+"];
  N04 -> N08[label="left:R+"];
  N06 -> N10[label="left:R+"];
  N07 -> N08[label="right:R"];
  N08 -> N09[label="right:R+"];
  N09 -> N10[label="right:R+"];
  N10 -> N11[label="operand:R+"];
  N11 -> N12[label="mu:R"];
  N12 -> N13[label="operand:R"];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_6(self) -> None:
        """test_fix_problems_6"""

        # This test shows that we can rewrite a division by a constant
        # into a multiplication.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # def hcs(): return HalfCauchy(1.0)
        # def norm(): return Normal(hcs() / 2.5, 1.0)

        one = bmg.add_constant(1.0)
        two = bmg.add_constant(2.5)
        hc = bmg.add_halfcauchy(one)
        hcs = bmg.add_sample(hc)
        q = bmg.add_division(hcs, two)
        norm = bmg.add_normal(q, one)
        bmg.add_sample(norm)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="1.0:R>=OH"];
  N1[label="HalfCauchy:R+>=R+"];
  N2[label="Sample:R+>=R+"];
  N3[label="2.5:R>=R+"];
  N4[label="/:M>=R+"];
  N5[label="Normal:R>=R"];
  N6[label="Sample:R>=R"];
  N0 -> N1[label="scale:R+"];
  N0 -> N5[label="sigma:R+"];
  N1 -> N2[label="operand:R+"];
  N2 -> N4[label="left:R+"];
  N3 -> N4[label="right:R+"];
  N4 -> N5[label="mu:R"];
  N5 -> N6[label="operand:R"];
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )
        expected = """
digraph "graph" {
  N00[label="1.0:R>=OH"];
  N01[label="1.0:R+>=OH"];
  N02[label="HalfCauchy:R+>=R+"];
  N03[label="Sample:R+>=R+"];
  N04[label="2.5:R>=R+"];
  N05[label="/:M>=R+"];
  N06[label="0.4:R+>=P"];
  N07[label="*:R+>=R+"];
  N08[label="ToReal:R>=R"];
  N09[label="Normal:R>=R"];
  N10[label="Sample:R>=R"];
  N11[label="0.4:R>=P"];
  N01 -> N02[label="scale:R+"];
  N01 -> N09[label="sigma:R+"];
  N02 -> N03[label="operand:R+"];
  N03 -> N05[label="left:R+"];
  N03 -> N07[label="left:R+"];
  N04 -> N05[label="right:R+"];
  N06 -> N07[label="right:R+"];
  N07 -> N08[label="operand:<=R"];
  N08 -> N09[label="mu:R"];
  N09 -> N10[label="operand:R"];
}
"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_7(self) -> None:
        """test_fix_problems_7"""

        # The problem here is that we have two uniform distributions that
        # we cannot turn into a flat distribution, and one we can. We therefore
        # expect that we will get two errors.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def foo1():
        #   return Uniform(0.0, 1.0) # OK
        # @rv def foo2():
        #   return Uniform(1.0, 2.0) # Bad
        # @rv def foo3():
        #   return Uniform(0.0, foo2()) # Bad

        zero = bmg.add_constant(0.0)
        one = bmg.add_constant(1.0)
        two = bmg.add_constant(2.0)
        foo1 = bmg.add_uniform(zero, one)
        bmg.add_sample(foo1)
        foo2 = bmg.add_uniform(one, two)
        foo2s = bmg.add_sample(foo2)
        foo3 = bmg.add_uniform(one, foo2s)
        bmg.add_sample(foo3)

        error_report = fix_problems(bmg)
        observed = str(error_report)
        expected = """
The model uses a Uniform operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
The model uses a Uniform operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_fix_problems_8(self) -> None:
        """test_fix_problems_8"""

        # This test shows that we can rewrite a chi2 into a gamma.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def hcs():
        #   return HalfCauchy(1.0)
        # @rv def chi2():
        #   return Chi2(hcs())

        one = bmg.add_constant(1.0)
        hc = bmg.add_halfcauchy(one)
        hcs = bmg.add_sample(hc)
        chi2 = bmg.add_chi2(hcs)
        bmg.add_sample(chi2)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="1.0:R>=OH"];
  N1[label="HalfCauchy:R+>=R+"];
  N2[label="Sample:R+>=R+"];
  N3[label="Chi2:R+>=R+"];
  N4[label="Sample:R+>=R+"];
  N0 -> N1[label="scale:R+"];
  N1 -> N2[label="operand:R+"];
  N2 -> N3[label="df:R+"];
  N3 -> N4[label="operand:R+"];
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="1.0:R>=OH"];
  N1[label="1.0:R+>=OH"];
  N2[label="HalfCauchy:R+>=R+"];
  N3[label="Sample:R+>=R+"];
  N4[label="Chi2:R+>=R+"];
  N5[label="0.5:R+>=P"];
  N6[label="*:R+>=R+"];
  N7[label="Gamma:R+>=R+"];
  N8[label="Sample:R+>=R+"];
  N1 -> N2[label="scale:R+"];
  N2 -> N3[label="operand:R+"];
  N3 -> N4[label="df:R+"];
  N3 -> N6[label="left:R+"];
  N5 -> N6[label="right:R+"];
  N5 -> N7[label="rate:R+"];
  N6 -> N7[label="concentration:R+"];
  N7 -> N8[label="operand:R+"];
}"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_9(self) -> None:
        """test_fix_problems_9"""

        # The problem we have here is that natural raised to bool
        # is not supported in BMG without converting both to
        # positive real, but natural raised to bool is plainly
        # natural. We generate an if-then-else.

        # @rv def berns():
        #   return Bernoulli(0.5)
        # @rv def nats():
        #   return Binomial(2, 0.5)
        # @rv def bino():
        #   return Binomial(nats() ** berns(), 0.5)

        self.maxDiff = None
        bmg = BMGraphBuilder()

        two = bmg.add_natural(2)
        half = bmg.add_probability(0.5)
        bern = bmg.add_bernoulli(half)
        berns = bmg.add_sample(bern)
        nat = bmg.add_binomial(two, half)
        nats = bmg.add_sample(nat)
        powr = bmg.add_power(nats, berns)
        bino = bmg.add_binomial(powr, half)
        bmg.add_sample(bino)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="2:N>=N"];
  N1[label="0.5:P>=P"];
  N2[label="Binomial:N>=N"];
  N3[label="Sample:N>=N"];
  N4[label="Bernoulli:B>=B"];
  N5[label="Sample:B>=B"];
  N6[label="**:M>=R+"];
  N7[label="Binomial:N>=N"];
  N8[label="Sample:N>=N"];
  N0 -> N2[label="count:N"];
  N1 -> N2[label="probability:P"];
  N1 -> N4[label="probability:P"];
  N1 -> N7[label="probability:P"];
  N2 -> N3[label="operand:N"];
  N3 -> N6[label="left:N"];
  N4 -> N5[label="operand:B"];
  N5 -> N6[label="right:B"];
  N6 -> N7[label="count:N"];
  N7 -> N8[label="operand:N"];
}"""

        self.assertEqual(expected.strip(), observed.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N00[label="2:N>=N"];
  N01[label="0.5:P>=P"];
  N02[label="Binomial:N>=N"];
  N03[label="Sample:N>=N"];
  N04[label="Bernoulli:B>=B"];
  N05[label="Sample:B>=B"];
  N06[label="**:M>=R+"];
  N07[label="1:N>=OH"];
  N08[label="if:N>=N"];
  N09[label="Binomial:N>=N"];
  N10[label="Sample:N>=N"];
  N11[label="1.0:R>=OH"];
  N00 -> N02[label="count:N"];
  N01 -> N02[label="probability:P"];
  N01 -> N04[label="probability:P"];
  N01 -> N09[label="probability:P"];
  N02 -> N03[label="operand:N"];
  N03 -> N06[label="left:N"];
  N03 -> N08[label="consequence:N"];
  N04 -> N05[label="operand:B"];
  N05 -> N06[label="right:B"];
  N05 -> N08[label="condition:B"];
  N07 -> N08[label="alternative:N"];
  N08 -> N09[label="count:N"];
  N09 -> N10[label="operand:N"];
}"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_10(self) -> None:
        """test_fix_problems_10"""

        # Demonstrate that we can rewrite 1-p for probability p into
        # complement(p) -- which is of type P -- instead of
        # add(1, negate(p)) which is of type R.

        # TODO: Also demonstrate that this works for 1-b
        # TODO: Get this working for the "not" operator, since 1-b
        # and "not b" are the same thing for bool b.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def beta():
        #   return Beta(2.0, 2.0)
        # @rv def bern():
        #   return Bernoulli(1 - beta()) # good!

        one = bmg.add_constant(1.0)
        two = bmg.add_constant(2.0)
        beta = bmg.add_beta(two, two)
        betas = bmg.add_sample(beta)
        negate = bmg.add_negate(betas)
        complement = bmg.add_addition(one, negate)
        bern = bmg.add_bernoulli(complement)
        bmg.add_sample(bern)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="1.0:R>=OH"];
  N1[label="2.0:R>=N"];
  N2[label="Beta:P>=P"];
  N3[label="Sample:P>=P"];
  N4[label="-:M>=R-"];
  N5[label="+:M>=R"];
  N6[label="Bernoulli:B>=B"];
  N7[label="Sample:B>=B"];
  N0 -> N5[label="left:OH"];
  N1 -> N2[label="alpha:R+"];
  N1 -> N2[label="beta:R+"];
  N2 -> N3[label="operand:P"];
  N3 -> N4[label="operand:R+"];
  N4 -> N5[label="right:R-"];
  N5 -> N6[label="probability:P"];
  N6 -> N7[label="operand:B"];
}
"""

        self.assertEqual(expected.strip(), observed.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="2.0:R>=N"];
  N1[label="1.0:R>=OH"];
  N2[label="2.0:R+>=N"];
  N3[label="Beta:P>=P"];
  N4[label="Sample:P>=P"];
  N5[label="-:M>=R-"];
  N6[label="+:M>=R"];
  N7[label="complement:P>=P"];
  N8[label="Bernoulli:B>=B"];
  N9[label="Sample:B>=B"];
  N1 -> N6[label="left:OH"];
  N2 -> N3[label="alpha:R+"];
  N2 -> N3[label="beta:R+"];
  N3 -> N4[label="operand:P"];
  N4 -> N5[label="operand:R+"];
  N4 -> N7[label="operand:P"];
  N5 -> N6[label="right:R-"];
  N7 -> N8[label="probability:P"];
  N8 -> N9[label="operand:B"];
}"""

        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_11(self) -> None:
        """test_fix_problems_11"""

        # Here we demonstrate that we treat the negative log of a
        # probability as a positive real.  (In a previous iteration
        # we generated a special negative log node, but now we can
        # do it directly without fixing up the graph.)

        # @rv def beta1():
        #   return Beta(2.0, 2.0)
        # @rv def beta2():
        #   return Beta(-beta1.log(), 2.0)

        self.maxDiff = None
        bmg = BMGraphBuilder()

        two = bmg.add_constant(2.0)
        beta1 = bmg.add_beta(two, two)
        beta1s = bmg.add_sample(beta1)
        logprob = bmg.add_log(beta1s)
        neglogprob = bmg.add_negate(logprob)
        beta2 = bmg.add_beta(neglogprob, two)
        bmg.add_sample(beta2)

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="2.0:R>=N"];
  N1[label="Beta:P>=P"];
  N2[label="Sample:P>=P"];
  N3[label="Log:R->=R-"];
  N4[label="-:R+>=R+"];
  N5[label="Beta:P>=P"];
  N6[label="Sample:P>=P"];
  N0 -> N1[label="alpha:R+"];
  N0 -> N1[label="beta:R+"];
  N0 -> N5[label="beta:R+"];
  N1 -> N2[label="operand:P"];
  N2 -> N3[label="operand:P"];
  N3 -> N4[label="operand:R-"];
  N4 -> N5[label="alpha:R+"];
  N5 -> N6[label="operand:P"];
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        error_report = fix_problems(bmg)
        self.assertEqual("", str(error_report).strip())

        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="2.0:R>=N"];
  N1[label="2.0:R+>=N"];
  N2[label="Beta:P>=P"];
  N3[label="Sample:P>=P"];
  N4[label="Log:R->=R-"];
  N5[label="-:R+>=R+"];
  N6[label="Beta:P>=P"];
  N7[label="Sample:P>=P"];
  N1 -> N2[label="alpha:R+"];
  N1 -> N2[label="beta:R+"];
  N1 -> N6[label="beta:R+"];
  N2 -> N3[label="operand:P"];
  N3 -> N4[label="operand:P"];
  N4 -> N5[label="operand:R-"];
  N5 -> N6[label="alpha:R+"];
  N6 -> N7[label="operand:P"];
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_12(self) -> None:
        """test_fix_problems_12"""

        # We flag impossible observations as errors.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def bern():
        #   return Bernoulli(0.5)
        # @rv def bino():
        #   return Binomial(2, 0.5)
        # @rv def norm():
        #   return Normal(0, 1)

        zero = bmg.add_constant(0.0)
        one = bmg.add_constant(1.0)
        two = bmg.add_constant(2.0)
        half = bmg.add_constant(0.5)
        bern = bmg.add_bernoulli(half)
        berns = bmg.add_sample(bern)
        bino = bmg.add_binomial(two, half)
        binos = bmg.add_sample(bino)
        norm = bmg.add_normal(zero, one)
        norms = bmg.add_sample(norm)

        bmg.add_observation(berns, -1.5)  # Bad
        bmg.add_observation(binos, 5.25)  # Bad
        bmg.add_observation(norms, True)  # OK; can be converted to 1.0

        error_report = fix_problems(bmg)
        observed = str(error_report)
        expected = """
A Bernoulli distribution is observed to have value -1.5 but only produces samples of type bool.
A Binomial distribution is observed to have value 5.25 but only produces samples of type natural.
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_fix_problems_13(self) -> None:
        """test_fix_problems_13"""

        # Observations of the wrong type are fixed up.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # @rv def bern():
        #   return Bernoulli(0.5)
        # @rv def bino():
        #   return Binomial(2, 0.5)
        # @rv def norm():
        #   return Normal(0, 1)

        zero = bmg.add_constant(0.0)
        one = bmg.add_constant(1.0)
        two = bmg.add_constant(2.0)
        half = bmg.add_constant(0.5)
        bern = bmg.add_bernoulli(half)
        berns = bmg.add_sample(bern)
        bino = bmg.add_binomial(two, half)
        binos = bmg.add_sample(bino)
        norm = bmg.add_normal(zero, one)
        norms = bmg.add_sample(norm)

        bmg.add_observation(berns, 0.0)  # Should be bool
        bmg.add_observation(binos, 5.0)  # Should be int
        bmg.add_observation(norms, True)  # Should be real

        error_report = fix_problems(bmg)
        self.assertEqual(str(error_report).strip(), "")
        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=True,
            edge_requirements=True,
        )

        # The observations have been converted to the correct types:
        expected = """
digraph "graph" {
  N00[label="1.0:R>=OH"];
  N01[label="2.0:R>=N"];
  N02[label="0.5:R>=P"];
  N03[label="0.5:P>=P"];
  N04[label="Bernoulli:B>=B"];
  N05[label="Sample:B>=B"];
  N06[label="Observation False:B>=B"];
  N07[label="2:N>=N"];
  N08[label="Binomial:N>=N"];
  N09[label="Sample:N>=N"];
  N10[label="Observation 5:N>=N"];
  N11[label="0.0:R>=Z"];
  N12[label="1.0:R+>=OH"];
  N13[label="Normal:R>=R"];
  N14[label="Sample:R>=R"];
  N15[label="Observation 1.0:R>=R"];
  N03 -> N04[label="probability:P"];
  N03 -> N08[label="probability:P"];
  N04 -> N05[label="operand:B"];
  N05 -> N06[label="operand:any"];
  N07 -> N08[label="count:N"];
  N08 -> N09[label="operand:N"];
  N09 -> N10[label="operand:any"];
  N11 -> N13[label="mu:R"];
  N12 -> N13[label="sigma:R+"];
  N13 -> N14[label="operand:R"];
  N14 -> N15[label="operand:any"];
}"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_problems_14(self) -> None:
        """test_fix_problems_14"""

        # Fixes for problems involving negative reals.

        self.maxDiff = None
        bmg = BMGraphBuilder()

        # Right now the only node we have of type negative real is
        # a constant; if we force a scenario where a negative real
        # constant is used in a context where a real is needed,
        # we generate a new real constant.

        m = bmg.add_neg_real(-1.0)
        s = bmg.add_pos_real(1.0)
        norm = bmg.add_normal(m, s)
        bmg.add_sample(norm)

        error_report = fix_problems(bmg)
        self.assertEqual(str(error_report).strip(), "")
        observed = to_dot(
            bmg,
            graph_types=True,
            inf_types=False,
            edge_requirements=True,
        )

        expected = """
digraph "graph" {
  N0[label="-1.0:R-"];
  N1[label="-1.0:R"];
  N2[label="1.0:R+"];
  N3[label="Normal:R"];
  N4[label="Sample:R"];
  N1 -> N3[label="mu:R"];
  N2 -> N3[label="sigma:R+"];
  N3 -> N4[label="operand:R"];
}
"""
        self.assertEqual(expected.strip(), observed.strip())
