# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for fix_problems.py"""
import unittest

from beanmachine.ppl.compiler.fix_problems import fix_problems
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from torch import tensor


class FixProblemsTest(unittest.TestCase):
    def test_fix_problems_1(self) -> None:
        """test_fix_problems_1"""

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

        observed = bmg.to_dot(True, False, True, True)
        expected = """
digraph "graph" {
  N00[label="1.0:T"];
  N01[label="2.0:T"];
  N02[label="0.5:T"];
  N03[label="Beta:P"];
  N04[label="Sample:P"];
  N05[label="*:M"];
  N06[label="Normal:R"];
  N07[label="Bernoulli:B"];
  N08[label="Sample:R"];
  N09[label="Sample:B"];
  N10[label="Query:M"];
  N00 -> N06[label="sigma:R+"];
  N01 -> N03[label="alpha:R+"];
  N01 -> N03[label="beta:R+"];
  N02 -> N05[label="left:P"];
  N03 -> N04[label="operand:P"];
  N04 -> N05[label="right:P"];
  N05 -> N06[label="mu:R"];
  N05 -> N07[label="probability:P"];
  N05 -> N10[label="operator:P"];
  N06 -> N08[label="operand:R"];
  N07 -> N09[label="operand:B"];
}"""
        self.assertEqual(observed.strip(), expected.strip())

        fix_problems(bmg)
        observed = bmg.to_dot(True, False, True, True)
        expected = """
digraph "graph" {
  N00[label="1.0:T"];
  N01[label="2.0:T"];
  N02[label="0.5:T"];
  N03[label="Beta:P"];
  N04[label="Sample:P"];
  N05[label="*:P"];
  N06[label="Normal:R"];
  N07[label="Bernoulli:B"];
  N08[label="Sample:R"];
  N09[label="Sample:B"];
  N10[label="Query:P"];
  N11[label="2.0:R+"];
  N12[label="0.5:P"];
  N13[label="ToReal:R"];
  N14[label="1.0:R+"];
  N03 -> N04[label="operand:P"];
  N04 -> N05[label="right:P"];
  N05 -> N07[label="probability:P"];
  N05 -> N10[label="operator:P"];
  N05 -> N13[label="operand:<=R"];
  N06 -> N08[label="operand:R"];
  N07 -> N09[label="operand:B"];
  N11 -> N03[label="alpha:R+"];
  N11 -> N03[label="beta:R+"];
  N12 -> N05[label="left:P"];
  N13 -> N06[label="mu:R"];
  N14 -> N06[label="sigma:R+"];
}
        """
        self.assertEqual(observed.strip(), expected.strip())

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

        observed = bmg.to_dot(True, False, True, True)
        expected = """
digraph "graph" {
  N0[label="1.0:T"];
  N1[label="0.5:T"];
  N2[label="Bernoulli:B"];
  N3[label="Sample:B"];
  N4[label="+:M"];
  N5[label="Normal:R"];
  N6[label="Binomial:N"];
  N7[label="Sample:R"];
  N8[label="Sample:N"];
  N0 -> N4[label="right:R+"];
  N1 -> N2[label="probability:P"];
  N1 -> N6[label="probability:P"];
  N2 -> N3[label="operand:B"];
  N3 -> N4[label="left:R+"];
  N3 -> N5[label="mu:R"];
  N3 -> N6[label="count:N"];
  N4 -> N5[label="sigma:R+"];
  N5 -> N7[label="operand:R"];
  N6 -> N8[label="operand:N"];
}
"""
        self.assertEqual(observed.strip(), expected.strip())

        fix_problems(bmg)
        observed = bmg.to_dot(True, False, True, True)
        expected = """
digraph "graph" {
  N00[label="1.0:T"];
  N01[label="0.5:T"];
  N02[label="Bernoulli:B"];
  N03[label="Sample:B"];
  N04[label="+:R+"];
  N05[label="Normal:R"];
  N06[label="Binomial:N"];
  N07[label="Sample:R"];
  N08[label="Sample:N"];
  N09[label="0.5:P"];
  N10[label="ToPosReal:R+"];
  N11[label="1.0:R+"];
  N12[label="ToReal:R"];
  N13[label="0:N"];
  N14[label="1:N"];
  N15[label="if:N"];
  N02 -> N03[label="operand:B"];
  N03 -> N10[label="operand:<=R+"];
  N03 -> N12[label="operand:<=R"];
  N03 -> N15[label="condition:B"];
  N04 -> N05[label="sigma:R+"];
  N05 -> N07[label="operand:R"];
  N06 -> N08[label="operand:N"];
  N09 -> N02[label="probability:P"];
  N09 -> N06[label="probability:P"];
  N10 -> N04[label="left:R+"];
  N11 -> N04[label="right:R+"];
  N12 -> N05[label="mu:R"];
  N13 -> N15[label="alternative:B"];
  N14 -> N15[label="consequence:B"];
  N15 -> N06[label="count:N"];
}
        """
        self.assertEqual(observed.strip(), expected.strip())

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
The probability of a Bernoulli is required to be a probability but is a tensor.
The probability of a Binomial is required to be a probability but is a positive real.
The sigma of a Normal is required to be a positive real but is a real.
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

        observed = bmg.to_dot(True, True, True, True)

        expected = """
digraph "graph" {
  N0[label="2:N>=N"];
  N1[label="0.5:P>=P"];
  N2[label="Bernoulli:B>=B"];
  N3[label="Sample:B>=B"];
  N4[label="Binomial:N>=N"];
  N5[label="Sample:N>=N"];
  N6[label="*:M>=N"];
  N7[label="Binomial:N>=N"];
  N8[label="Sample:N>=N"];
  N0 -> N4[label="count:N"];
  N1 -> N2[label="probability:P"];
  N1 -> N4[label="probability:P"];
  N1 -> N7[label="probability:P"];
  N2 -> N3[label="operand:B"];
  N3 -> N6[label="left:B"];
  N4 -> N5[label="operand:N"];
  N5 -> N6[label="right:N"];
  N6 -> N7[label="count:N"];
  N7 -> N8[label="operand:N"];
}"""

        self.assertEqual(observed.strip(), expected.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = bmg.to_dot(True, True, True, True)

        expected = """

digraph "graph" {
  N00[label="2:N>=N"];
  N01[label="0.5:P>=P"];
  N02[label="Bernoulli:B>=B"];
  N03[label="Sample:B>=B"];
  N04[label="Binomial:N>=N"];
  N05[label="Sample:N>=N"];
  N06[label="*:M>=N"];
  N07[label="Binomial:N>=N"];
  N08[label="Sample:N>=N"];
  N09[label="0:N>=B"];
  N10[label="if:N>=N"];
  N00 -> N04[label="count:N"];
  N01 -> N02[label="probability:P"];
  N01 -> N04[label="probability:P"];
  N01 -> N07[label="probability:P"];
  N02 -> N03[label="operand:B"];
  N03 -> N06[label="left:B"];
  N03 -> N10[label="condition:B"];
  N04 -> N05[label="operand:N"];
  N05 -> N06[label="right:N"];
  N05 -> N10[label="consequence:N"];
  N07 -> N08[label="operand:N"];
  N09 -> N10[label="alternative:N"];
  N10 -> N07[label="count:N"];
}
                """

        self.assertEqual(observed.strip(), expected.strip())

    def test_fix_problems_5(self) -> None:
        """test_fix_problems_5"""

        # This test has some problems that cannot be fixed.
        #
        # * No support for power yet
        # * No support for division yet
        # * No support for log yet

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
        expected = """
The model uses a ** operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Log.
The model uses a / operation unsupported by Bean Machine Graph.
The unsupported node is the right of a **.
The model uses a Log operation unsupported by Bean Machine Graph.
The unsupported node is the mu of a Normal.
"""
        self.assertEqual(observed.strip(), expected.strip())

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

        observed = bmg.to_dot(True, True, True, True)

        expected = """
digraph "graph" {
  N0[label="1.0:R>=B"];
  N1[label="2.5:R>=R+"];
  N2[label="HalfCauchy:R+>=R+"];
  N3[label="Sample:R+>=R+"];
  N4[label="/:M>=R+"];
  N5[label="Normal:R>=R"];
  N6[label="Sample:R>=R"];
  N0 -> N2[label="scale:R+"];
  N0 -> N5[label="sigma:R+"];
  N1 -> N4[label="right:R+"];
  N2 -> N3[label="operand:R+"];
  N3 -> N4[label="left:R+"];
  N4 -> N5[label="mu:R"];
  N5 -> N6[label="operand:R"];
}
"""

        self.assertEqual(observed.strip(), expected.strip())

        error_report = fix_problems(bmg)

        self.assertEqual("", str(error_report).strip())

        observed = bmg.to_dot(True, True, True, True)

        expected = """
digraph "graph" {
  N00[label="1.0:R>=B"];
  N01[label="2.5:R>=R+"];
  N02[label="HalfCauchy:R+>=R+"];
  N03[label="Sample:R+>=R+"];
  N04[label="/:M>=R+"];
  N05[label="Normal:R>=R"];
  N06[label="Sample:R>=R"];
  N07[label="0.4:R>=P"];
  N08[label="*:R+>=R+"];
  N09[label="1.0:R+>=B"];
  N10[label="0.4:R+>=P"];
  N11[label="ToReal:R>=R"];
  N01 -> N04[label="right:R+"];
  N02 -> N03[label="operand:R+"];
  N03 -> N04[label="left:R+"];
  N03 -> N08[label="left:R+"];
  N05 -> N06[label="operand:R"];
  N08 -> N11[label="operand:<=R"];
  N09 -> N02[label="scale:R+"];
  N09 -> N05[label="sigma:R+"];
  N10 -> N08[label="right:R+"];
  N11 -> N05[label="mu:R"];
}
"""

        self.assertEqual(observed.strip(), expected.strip())

    def test_fix_problems_7(self) -> None:
        """test_fix_problems_7"""

        # This test has a problem that cannot be fixed yet, but
        # will be fixable soon; when it is, update this test to
        # show the fixed and unfixed problem.

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
The model uses a Uniform operation unsupported by Bean Machine Graph.
The unsupported node is the operand of a Sample.
"""
        self.assertEqual(observed.strip(), expected.strip())
