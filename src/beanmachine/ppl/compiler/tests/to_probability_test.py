# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.compiler.hint import log1mexp
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli, Beta


# If we use a positive real valued *operator* in a context where
# a probability is required, we allow it. But we don't allow constants.
#
# For example, if we have a probability divided by two, that's still a
# probability. But adding it to another probability results in a positive
# real even though we know it is still between 0.0 and 1.0.
#
# What we do in this situation is automatically insert a "to probability"
# operator that coerces the positive real to a probability.


@bm.random_variable
def beta(n):
    return Beta(2.0, 2.0)


@bm.random_variable
def flip():
    return Bernoulli(beta(0) * 0.5 + 0.5)


# However, we should still reject constants that are out of bounds.


@bm.random_variable
def bad_flip():
    return Bernoulli(2.5)


# Similarly for log-probabilities which are negative reals.


@bm.functional
def to_neg_real():
    pr1 = beta(1) * 0.5 + 0.5  # positive real
    pr2 = beta(2) * 0.5 + 0.5  # positive real
    lg1 = pr1.log()  # real
    lg2 = pr2.log()  # real
    # Because we think pr1 and pr2 are positive reals instead of probabilities,
    # we also think that lg1 and lg2 are reals instead of negative reals.
    inv = log1mexp(lg1 + lg2)  # needs a negative real
    # We should insert a TO_NEG_REAL node on the sum above.
    return inv


class ToProbabilityTest(unittest.TestCase):
    def test_to_probability_1(self) -> None:

        self.maxDiff = None
        bmg = BMGInference()
        observed = bmg.to_dot([flip()], {})

        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=0.5];
  N04[label="*"];
  N05[label=ToPosReal];
  N06[label=0.5];
  N07[label="+"];
  N08[label=ToProb];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N04;
  N03 -> N04;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.assertEqual(observed.strip(), expected.strip())

    def test_to_probability_2(self) -> None:

        self.maxDiff = None
        bmg = BMGInference()

        # TODO: Raise a better error than a generic ValueError
        with self.assertRaises(ValueError) as ex:
            bmg.infer([bad_flip()], {}, 10)
        self.assertEqual(
            str(ex.exception),
            "The probability of a Bernoulli is required to be a"
            + " probability but is a positive real.",
        )

    def test_to_neg_real_1(self) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot([to_neg_real()], {})
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=0.5];
  N05[label="*"];
  N06[label=ToPosReal];
  N07[label=0.5];
  N08[label="+"];
  N09[label=Log];
  N10[label="*"];
  N11[label=ToPosReal];
  N12[label="+"];
  N13[label=Log];
  N14[label="+"];
  N15[label=ToNegReal];
  N16[label=Log1mexp];
  N17[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N05;
  N03 -> N10;
  N04 -> N05;
  N04 -> N10;
  N05 -> N06;
  N06 -> N08;
  N07 -> N08;
  N07 -> N12;
  N08 -> N09;
  N09 -> N14;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
