# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
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
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip():
    return Bernoulli(beta() * 0.5 + 0.5)


# However, we should still reject constants that are out of bounds.


@bm.random_variable
def bad_flip():
    return Bernoulli(2.5)


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
