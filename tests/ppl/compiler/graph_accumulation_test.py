# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for bm_to_bmg.py"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.gen_bmg_graph import to_bmg_graph
from beanmachine.ppl.compiler.runtime import BMGRuntime
from torch import tensor
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Chi2,
    Gamma,
    HalfCauchy,
    Normal,
    StudentT,
    Uniform,
)


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


# These are cases where we just have either a straightforward sample from
# a distribution parameterized with constants, or a distribution parameterized
# with a sample from another distribution.
#
# * No arithmetic
# * No interesting type conversions
# * No use of a sample as an index.
#


@bm.random_variable
def flip_straight_constant():
    return Bernoulli(tensor(0.5))


@bm.random_variable
def flip_logit_constant():
    logits = tensor(-2.0)
    return Bernoulli(logits=logits)


@bm.random_variable
def standard_normal():
    return Normal(0.0, 1.0)


@bm.random_variable
def flip_logit_normal():
    logits = standard_normal()
    return Bernoulli(logits=logits)


@bm.random_variable
def beta_constant():
    return Beta(1.0, 1.0)


@bm.random_variable
def hc(i):
    return HalfCauchy(1.0)


@bm.random_variable
def beta_hc():
    return Beta(hc(1), hc(2))


@bm.random_variable
def student_t():
    return StudentT(hc(1), standard_normal(), hc(2))


@bm.random_variable
def bin_constant():
    return Binomial(3, 0.5)


@bm.random_variable
def gamma():
    return Gamma(1.0, 2.0)


@bm.random_variable
def flat():
    return Uniform(0.0, 1.0)


@bm.random_variable
def chi2():
    return Chi2(8.0)


expected_bmg_1 = """
0: CONSTANT(probability 0.5) (out nodes: 1, 22)
1: BERNOULLI(0) (out nodes: 2)
2: SAMPLE(1) (out nodes: ) queried
3: CONSTANT(probability 0.119203) (out nodes: 4)
4: BERNOULLI(3) (out nodes: 5)
5: SAMPLE(4) (out nodes: ) queried
6: CONSTANT(real 0) (out nodes: 8)
7: CONSTANT(positive real 1) (out nodes: 8, 12, 12, 14, 25)
8: NORMAL(6, 7) (out nodes: 9)
9: SAMPLE(8) (out nodes: 10, 19) queried
10: BERNOULLI_LOGIT(9) (out nodes: 11)
11: SAMPLE(10) (out nodes: ) queried
12: BETA(7, 7) (out nodes: 13)
13: SAMPLE(12) (out nodes: ) queried
14: HALF_CAUCHY(7) (out nodes: 15, 16)
15: SAMPLE(14) (out nodes: 17, 19) queried
16: SAMPLE(14) (out nodes: 17, 19) queried
17: BETA(15, 16) (out nodes: 18)
18: SAMPLE(17) (out nodes: ) queried
19: STUDENT_T(15, 9, 16) (out nodes: 20)
20: SAMPLE(19) (out nodes: ) queried
21: CONSTANT(natural 3) (out nodes: 22)
22: BINOMIAL(21, 0) (out nodes: 23)
23: SAMPLE(22) (out nodes: ) queried
24: CONSTANT(positive real 2) (out nodes: 25)
25: GAMMA(7, 24) (out nodes: 26)
26: SAMPLE(25) (out nodes: ) queried
27: FLAT() (out nodes: 28)
28: SAMPLE(27) (out nodes: ) queried
29: CONSTANT(positive real 4) (out nodes: 31)
30: CONSTANT(positive real 0.5) (out nodes: 31)
31: GAMMA(29, 30) (out nodes: 32)
32: SAMPLE(31) (out nodes: ) queried
"""

# These are cases where we have a type conversion on a sample.


@bm.random_variable
def normal_from_bools():
    # Converts Boolean to real, positive real
    # This is of course dubious as we would not typically
    # expect the standard deviation to be zero or one, but
    # it illustrates that the type conversion works.
    # TODO: Consider adding a warning for conversion from
    # TODO: bool to positive real.
    return Normal(flip_straight_constant(), flip_straight_constant())


@bm.random_variable
def binomial_from_bools():
    # Converts Boolean to natural and probability
    return Binomial(flip_straight_constant(), flip_straight_constant())


expected_bmg_2 = """
0: CONSTANT(probability 0.5) (out nodes: 1)
1: BERNOULLI(0) (out nodes: 2)
2: SAMPLE(1) (out nodes: 3, 4, 9, 12)
3: TO_REAL(2) (out nodes: 5)
4: TO_POS_REAL(2) (out nodes: 5)
5: NORMAL(3, 4) (out nodes: 6)
6: SAMPLE(5) (out nodes: ) queried
7: CONSTANT(natural 1) (out nodes: 9)
8: CONSTANT(natural 0) (out nodes: 9)
9: IF_THEN_ELSE(2, 7, 8) (out nodes: 13)
10: CONSTANT(probability 1) (out nodes: 12)
11: CONSTANT(probability 1e-10) (out nodes: 12)
12: IF_THEN_ELSE(2, 10, 11) (out nodes: 13)
13: BINOMIAL(9, 12) (out nodes: 14)
14: SAMPLE(13) (out nodes: ) queried
"""


# Here we multiply a bool by a natural, and then use that as a natural.
# This cannot be turned into a BMG that uses multiplication because
# there is no multiplication defined on naturals or bools; the best
# we could do as a multiplication is to turn both into a positive real
# and multiply those.  But we *can* turn this into an if-then-else
# that takes a bool and returns either the given natural or zero,
# so that's what we'll do.


@bm.random_variable
def bool_times_natural():
    return Binomial(bin_constant() * flip_straight_constant(), 0.5)


expected_bmg_3 = """
0: CONSTANT(natural 3) (out nodes: 2)
1: CONSTANT(probability 0.5) (out nodes: 2, 4, 8)
2: BINOMIAL(0, 1) (out nodes: 3)
3: SAMPLE(2) (out nodes: 7)
4: BERNOULLI(1) (out nodes: 5)
5: SAMPLE(4) (out nodes: 7)
6: CONSTANT(natural 0) (out nodes: 7)
7: IF_THEN_ELSE(5, 3, 6) (out nodes: 8)
8: BINOMIAL(7, 1) (out nodes: 9)
9: SAMPLE(8) (out nodes: ) queried
"""

# Tests for math functions


@bm.random_variable
def math1():
    # log(R+) -> R
    # exp(R+) -> R+
    return Normal(hc(0).log(), hc(1).exp())


@bm.random_variable
def math2():
    # R+ ** R+ -> R+
    return HalfCauchy(hc(2) ** hc(3))


@bm.random_variable
def math3():
    # PHI
    return Bernoulli(Normal(0.0, 1.0).cdf(hc(4)))


@bm.random_variable
def math4():
    # PHI, alternative syntax
    # TODO: Add a test where the value passed to cdf is a named argument.
    return Bernoulli(Normal.cdf(Normal(0.0, 1.0), hc(4)))


expected_bmg_4 = """
0: CONSTANT(positive real 1) (out nodes: 1)
1: HALF_CAUCHY(0) (out nodes: 2, 3, 8, 9, 13)
2: SAMPLE(1) (out nodes: 4)
3: SAMPLE(1) (out nodes: 5)
4: LOG(2) (out nodes: 6)
5: EXP(3) (out nodes: 6)
6: NORMAL(4, 5) (out nodes: 7)
7: SAMPLE(6) (out nodes: ) queried
8: SAMPLE(1) (out nodes: 10)
9: SAMPLE(1) (out nodes: 10)
10: POW(8, 9) (out nodes: 11)
11: HALF_CAUCHY(10) (out nodes: 12)
12: SAMPLE(11) (out nodes: ) queried
13: SAMPLE(1) (out nodes: 14)
14: TO_REAL(13) (out nodes: 15)
15: PHI(14) (out nodes: 16)
16: BERNOULLI(15) (out nodes: 17)
17: SAMPLE(16) (out nodes: ) queried
"""

# Demonstrate that we generate 1-p as a complement


@bm.random_variable
def flip_complement():
    return Bernoulli(1.0 - beta_constant())


expected_bmg_5 = """
0: CONSTANT(positive real 1) (out nodes: 1, 1)
1: BETA(0, 0) (out nodes: 2)
2: SAMPLE(1) (out nodes: 3)
3: COMPLEMENT(2) (out nodes: 4)
4: BERNOULLI(3) (out nodes: 5)
5: SAMPLE(4) (out nodes: ) queried
"""


# Demonstrate that we generate -log(prob) as a positive real.


@bm.random_variable
def beta_neg_log():
    return Beta(-beta_constant().log(), 1.0)


expected_bmg_6 = """
0: CONSTANT(positive real 1) (out nodes: 1, 1, 5)
1: BETA(0, 0) (out nodes: 2)
2: SAMPLE(1) (out nodes: 3)
3: LOG(2) (out nodes: 4)
4: NEGATE(3) (out nodes: 5)
5: BETA(4, 0) (out nodes: 6)
6: SAMPLE(5) (out nodes: ) queried
"""

# Demonstrate that identity additions and multiplications
# are removed from the graph.  Here we are computing
# 0 + 0 * hc(0) + 1 * hc(1) + 2 * hc(2)
# but as you can see, in the final program we generate
# the code as though we had written hc(1) + 2 * hc(2).
#
# TODO: However, note that we still do emit a sample
# for hc(0) into the graph, even though it is unused.
# We might consider trimming sample operations which
# are ancestors of no observation or query.


@bm.random_variable
def beta_eliminate_identities():
    s = 0.0
    for i in [0, 1, 2]:
        s = s + i * hc(i)
    return Beta(s, 4.0)


expected_bmg_7 = """
digraph "graph" {
  N0[label="1"];
  N1[label="HalfCauchy"];
  N2[label="~"];
  N3[label="~"];
  N4[label="~"];
  N5[label="2"];
  N6[label="*"];
  N7[label="+"];
  N8[label="4"];
  N9[label="Beta"];
  N10[label="~"];
  N0 -> N1;
  N1 -> N2;
  N1 -> N3;
  N1 -> N4;
  N3 -> N7;
  N4 -> N6;
  N5 -> N6;
  N6 -> N7;
  N7 -> N9;
  N8 -> N9;
  N9 -> N10;
  Q0[label="Query"];
  N10 -> Q0;
}
"""


class GraphAccumulationTests(unittest.TestCase):
    def test_accumulate_simple_distributions(self) -> None:
        self.maxDiff = None
        queries = [
            flip_straight_constant(),
            flip_logit_constant(),
            standard_normal(),
            flip_logit_normal(),
            beta_constant(),
            hc(1),
            hc(2),
            beta_hc(),
            student_t(),
            bin_constant(),
            gamma(),
            flat(),
            chi2(),
        ]

        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_1))

    def test_accumulate_bool_conversions(self) -> None:
        self.maxDiff = None
        queries = [normal_from_bools(), binomial_from_bools()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_2))

    def test_accumulate_bool_nat_mult(self) -> None:
        self.maxDiff = None
        queries = [bool_times_natural()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_3))

    def test_accumulate_math(self) -> None:
        self.maxDiff = None
        queries = [math1(), math2(), math3()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_4))

        # Try with a different version of CDF syntax.
        queries = [math1(), math2(), math4()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_4))

    def test_accumulate_complement(self) -> None:
        self.maxDiff = None
        queries = [flip_complement()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_5))

    def test_accumulate_neg_log(self) -> None:
        self.maxDiff = None
        queries = [beta_neg_log()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg_6))

    def test_accumulate_eliminate_identities(self) -> None:
        self.maxDiff = None
        # TODO: We end up with an extraneous zero addend in the
        # sum; eliminate that.
        queries = [beta_eliminate_identities()]
        bmg = BMGRuntime().accumulate_graph(queries, {})
        observed = to_bmg_graph(bmg).graph.to_dot()
        self.assertEqual(expected_bmg_7.strip(), observed.strip())
