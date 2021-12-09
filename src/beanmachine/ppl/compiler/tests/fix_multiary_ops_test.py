# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def sum_1():
    return norm(0) + norm(1) + norm(2)


@bm.functional
def sum_2():
    return norm(3) + norm(4) + norm(5)


@bm.functional
def sum_3():
    return sum_1() + 5.0


@bm.functional
def sum_4():
    return sum_1() + sum_2()


@bm.functional
def mult_1():
    return norm(0) * norm(1) * norm(2)


@bm.functional
def mult_2():
    return norm(3) * norm(4) * norm(5)


@bm.functional
def mult_3():
    return mult_1() * 5.0


@bm.functional
def mult_4():
    return mult_1() * mult_2()


class FixMultiaryOperatorTest(unittest.TestCase):
    def test_fix_multiary_addition_1(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [sum_3(), sum_4()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before optimization

        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label="+"];
  N06[label=Sample];
  N07[label="+"];
  N08[label=5.0];
  N09[label="+"];
  N10[label=Query];
  N11[label=Sample];
  N12[label=Sample];
  N13[label="+"];
  N14[label=Sample];
  N15[label="+"];
  N16[label="+"];
  N17[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N06;
  N02 -> N11;
  N02 -> N12;
  N02 -> N14;
  N03 -> N05;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N07 -> N09;
  N07 -> N16;
  N08 -> N09;
  N09 -> N10;
  N11 -> N13;
  N12 -> N13;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After optimization:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label="+"];
  N07[label=5.0];
  N08[label="+"];
  N09[label=Query];
  N10[label=Sample];
  N11[label=Sample];
  N12[label=Sample];
  N13[label="+"];
  N14[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N02 -> N10;
  N02 -> N11;
  N02 -> N12;
  N03 -> N06;
  N04 -> N06;
  N05 -> N06;
  N06 -> N08;
  N06 -> N13;
  N07 -> N08;
  N08 -> N09;
  N10 -> N13;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_fix_multiary_multiplication(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [mult_3(), mult_4()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before optimization
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label="*"];
  N06[label=Sample];
  N07[label="*"];
  N08[label=5.0];
  N09[label="*"];
  N10[label=Query];
  N11[label=Sample];
  N12[label=Sample];
  N13[label="*"];
  N14[label=Sample];
  N15[label="*"];
  N16[label="*"];
  N17[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N06;
  N02 -> N11;
  N02 -> N12;
  N02 -> N14;
  N03 -> N05;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N07 -> N09;
  N07 -> N16;
  N08 -> N09;
  N09 -> N10;
  N11 -> N13;
  N12 -> N13;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After optimization:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label="*"];
  N07[label=5.0];
  N08[label="*"];
  N09[label=Query];
  N10[label=Sample];
  N11[label=Sample];
  N12[label=Sample];
  N13[label="*"];
  N14[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N02 -> N10;
  N02 -> N11;
  N02 -> N12;
  N03 -> N06;
  N04 -> N06;
  N05 -> N06;
  N06 -> N08;
  N06 -> N13;
  N07 -> N08;
  N08 -> N09;
  N10 -> N13;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
