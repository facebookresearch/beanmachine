# Copyright (c) Meta Platforms, Inc. and its affiliates.
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
Node 0 type 1 parents [ ] children [ 1 22 ] probability 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ ] boolean 0
Node 3 type 1 parents [ ] children [ 4 ] probability 0.119203
Node 4 type 2 parents [ 3 ] children [ 5 ] unknown
Node 5 type 3 parents [ 4 ] children [ ] boolean 0
Node 6 type 1 parents [ ] children [ 8 ] real 0
Node 7 type 1 parents [ ] children [ 8 12 12 14 25 ] positive real 1
Node 8 type 2 parents [ 6 7 ] children [ 9 ] unknown
Node 9 type 3 parents [ 8 ] children [ 10 19 ] real 0
Node 10 type 2 parents [ 9 ] children [ 11 ] unknown
Node 11 type 3 parents [ 10 ] children [ ] boolean 0
Node 12 type 2 parents [ 7 7 ] children [ 13 ] unknown
Node 13 type 3 parents [ 12 ] children [ ] probability 1e-10
Node 14 type 2 parents [ 7 ] children [ 15 16 ] unknown
Node 15 type 3 parents [ 14 ] children [ 17 19 ] positive real 1e-10
Node 16 type 3 parents [ 14 ] children [ 17 19 ] positive real 1e-10
Node 17 type 2 parents [ 15 16 ] children [ 18 ] unknown
Node 18 type 3 parents [ 17 ] children [ ] probability 1e-10
Node 19 type 2 parents [ 15 9 16 ] children [ 20 ] unknown
Node 20 type 3 parents [ 19 ] children [ ] real 0
Node 21 type 1 parents [ ] children [ 22 ] natural 3
Node 22 type 2 parents [ 21 0 ] children [ 23 ] unknown
Node 23 type 3 parents [ 22 ] children [ ] natural 0
Node 24 type 1 parents [ ] children [ 25 ] positive real 2
Node 25 type 2 parents [ 7 24 ] children [ 26 ] unknown
Node 26 type 3 parents [ 25 ] children [ ] positive real 1e-10
Node 27 type 2 parents [ ] children [ 28 ] unknown
Node 28 type 3 parents [ 27 ] children [ ] probability 1e-10
Node 29 type 1 parents [ ] children [ 31 ] positive real 4
Node 30 type 1 parents [ ] children [ 31 ] positive real 0.5
Node 31 type 2 parents [ 29 30 ] children [ 32 ] unknown
Node 32 type 3 parents [ 31 ] children [ ] positive real 1e-10
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
Node 0 type 1 parents [ ] children [ 1 ] probability 0.5
Node 1 type 2 parents [ 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 4 9 12 ] boolean 0
Node 3 type 3 parents [ 2 ] children [ 5 ] real 0
Node 4 type 3 parents [ 2 ] children [ 5 ] positive real 1e-10
Node 5 type 2 parents [ 3 4 ] children [ 6 ] unknown
Node 6 type 3 parents [ 5 ] children [ ] real 0
Node 7 type 1 parents [ ] children [ 9 ] natural 1
Node 8 type 1 parents [ ] children [ 9 ] natural 0
Node 9 type 3 parents [ 2 7 8 ] children [ 13 ] natural 1
Node 10 type 1 parents [ ] children [ 12 ] probability 1
Node 11 type 1 parents [ ] children [ 12 ] probability 1e-10
Node 12 type 3 parents [ 2 10 11 ] children [ 13 ] probability 1
Node 13 type 2 parents [ 9 12 ] children [ 14 ] unknown
Node 14 type 3 parents [ 13 ] children [ ] natural 0
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
Node 0 type 1 parents [ ] children [ 2 ] natural 3
Node 1 type 1 parents [ ] children [ 2 4 8 ] probability 0.5
Node 2 type 2 parents [ 0 1 ] children [ 3 ] unknown
Node 3 type 3 parents [ 2 ] children [ 7 ] natural 0
Node 4 type 2 parents [ 1 ] children [ 5 ] unknown
Node 5 type 3 parents [ 4 ] children [ 7 ] boolean 0
Node 6 type 1 parents [ ] children [ 7 ] natural 0
Node 7 type 3 parents [ 5 3 6 ] children [ 8 ] natural 0
Node 8 type 2 parents [ 7 1 ] children [ 9 ] unknown
Node 9 type 3 parents [ 8 ] children [ ] natural 0
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
    return Bernoulli(Normal.cdf(Normal(0.0, 1.0), hc(4)))


expected_bmg_4 = """
Node 0 type 1 parents [ ] children [ 1 ] positive real 1
Node 1 type 2 parents [ 0 ] children [ 2 3 8 9 13 ] unknown
Node 2 type 3 parents [ 1 ] children [ 4 ] positive real 1e-10
Node 3 type 3 parents [ 1 ] children [ 5 ] positive real 1e-10
Node 4 type 3 parents [ 2 ] children [ 6 ] real 0
Node 5 type 3 parents [ 3 ] children [ 6 ] positive real 1e-10
Node 6 type 2 parents [ 4 5 ] children [ 7 ] unknown
Node 7 type 3 parents [ 6 ] children [ ] real 0
Node 8 type 3 parents [ 1 ] children [ 10 ] positive real 1e-10
Node 9 type 3 parents [ 1 ] children [ 10 ] positive real 1e-10
Node 10 type 3 parents [ 8 9 ] children [ 11 ] positive real 1e-10
Node 11 type 2 parents [ 10 ] children [ 12 ] unknown
Node 12 type 3 parents [ 11 ] children [ ] positive real 1e-10
Node 13 type 3 parents [ 1 ] children [ 14 ] positive real 1e-10
Node 14 type 3 parents [ 13 ] children [ 15 ] real 0
Node 15 type 3 parents [ 14 ] children [ 16 ] probability 1e-10
Node 16 type 2 parents [ 15 ] children [ 17 ] unknown
Node 17 type 3 parents [ 16 ] children [ ] boolean 0
"""

# Demonstrate that we generate 1-p as a complement


@bm.random_variable
def flip_complement():
    return Bernoulli(1.0 - beta_constant())


expected_bmg_5 = """
Node 0 type 1 parents [ ] children [ 1 1 ] positive real 1
Node 1 type 2 parents [ 0 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 ] probability 1e-10
Node 3 type 3 parents [ 2 ] children [ 4 ] probability 1e-10
Node 4 type 2 parents [ 3 ] children [ 5 ] unknown
Node 5 type 3 parents [ 4 ] children [ ] boolean 0
"""


# Demonstrate that we generate -log(prob) as a positive real.


@bm.random_variable
def beta_neg_log():
    return Beta(-beta_constant().log(), 1.0)


expected_bmg_6 = """
Node 0 type 1 parents [ ] children [ 1 1 5 ] positive real 1
Node 1 type 2 parents [ 0 0 ] children [ 2 ] unknown
Node 2 type 3 parents [ 1 ] children [ 3 ] probability 1e-10
Node 3 type 3 parents [ 2 ] children [ 4 ] negative real -1e-10
Node 4 type 3 parents [ 3 ] children [ 5 ] positive real 1e-10
Node 5 type 2 parents [ 4 0 ] children [ 6 ] unknown
Node 6 type 3 parents [ 5 ] children [ ] probability 1e-10
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
