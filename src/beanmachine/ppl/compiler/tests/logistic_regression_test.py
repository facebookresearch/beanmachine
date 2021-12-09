# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end test of realistic logistic regression model"""
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Normal


# We have N points with K coordinates each classified into one
# of two categories: red or blue. There is a line separating
# the sets of points; the idea is to deduce the most likely
# parameters of that line.  Parameters are beta(0), beta(1)
# and beta(2); line is y = (-b1/b2) x - (b0/b2).
#
# We have three parameters to define the line instead of two because
# these parameters also define how "mixed" the points are when close
# to the line.

# Points are generated so that posteriors should be
# centered on beta(0) around -1.0, beta(1) around 2.0,
# beta(2) around -3.0

N = 8
K = 2
X = [
    [1.0000, 7.6483, 5.6988],
    [1.0000, -6.2928, 1.1692],
    [1.0000, 1.6583, -4.7142],
    [1.0000, -7.7588, 7.9859],
    [1.0000, -1.2421, 5.4628],
    [1.0000, 6.4529, 2.3994],
    [1.0000, -4.9269, 7.8679],
    [1.0000, 4.2130, 2.6175],
]

# Classifications of those N points into two buckets:

red = tensor(0.0)
blue = tensor(1.0)
Y = [red, red, blue, red, red, blue, red, blue]


@bm.random_variable
def beta(k):  # k is 0 to K
    return Normal(0.0, 1.0)


@bm.random_variable
def y(n):  # n is 0 to N-1
    mu = X[n][0] * beta(0) + X[n][1] * beta(1) + X[n][2] * beta(2)
    return Bernoulli(logits=mu)


queries = [beta(0), beta(1), beta(2)]
observations = {
    y(0): Y[0],
    y(1): Y[1],
    y(2): Y[2],
    y(3): Y[3],
    y(4): Y[4],
    y(5): Y[5],
    y(6): Y[6],
    y(7): Y[7],
}


expected_dot = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label=7.6483];
  N07[label="*"];
  N08[label=5.6988];
  N09[label="*"];
  N10[label="+"];
  N11[label="Bernoulli(logits)"];
  N12[label=Sample];
  N13[label="Observation False"];
  N14[label=-6.2928];
  N15[label="*"];
  N16[label=1.1692];
  N17[label="*"];
  N18[label="+"];
  N19[label="Bernoulli(logits)"];
  N20[label=Sample];
  N21[label="Observation False"];
  N22[label=1.6583];
  N23[label="*"];
  N24[label=-4.7142];
  N25[label="*"];
  N26[label="+"];
  N27[label="Bernoulli(logits)"];
  N28[label=Sample];
  N29[label="Observation True"];
  N30[label=-7.7588];
  N31[label="*"];
  N32[label=7.9859];
  N33[label="*"];
  N34[label="+"];
  N35[label="Bernoulli(logits)"];
  N36[label=Sample];
  N37[label="Observation False"];
  N38[label=-1.2421];
  N39[label="*"];
  N40[label=5.4628];
  N41[label="*"];
  N42[label="+"];
  N43[label="Bernoulli(logits)"];
  N44[label=Sample];
  N45[label="Observation False"];
  N46[label=6.4529];
  N47[label="*"];
  N48[label=2.3994];
  N49[label="*"];
  N50[label="+"];
  N51[label="Bernoulli(logits)"];
  N52[label=Sample];
  N53[label="Observation True"];
  N54[label=-4.9269];
  N55[label="*"];
  N56[label=7.8679];
  N57[label="*"];
  N58[label="+"];
  N59[label="Bernoulli(logits)"];
  N60[label=Sample];
  N61[label="Observation False"];
  N62[label=4.213];
  N63[label="*"];
  N64[label=2.6175];
  N65[label="*"];
  N66[label="+"];
  N67[label="Bernoulli(logits)"];
  N68[label=Sample];
  N69[label="Observation True"];
  N70[label=Query];
  N71[label=Query];
  N72[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N03 -> N10;
  N03 -> N18;
  N03 -> N26;
  N03 -> N34;
  N03 -> N42;
  N03 -> N50;
  N03 -> N58;
  N03 -> N66;
  N03 -> N70;
  N04 -> N07;
  N04 -> N15;
  N04 -> N23;
  N04 -> N31;
  N04 -> N39;
  N04 -> N47;
  N04 -> N55;
  N04 -> N63;
  N04 -> N71;
  N05 -> N09;
  N05 -> N17;
  N05 -> N25;
  N05 -> N33;
  N05 -> N41;
  N05 -> N49;
  N05 -> N57;
  N05 -> N65;
  N05 -> N72;
  N06 -> N07;
  N07 -> N10;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N14 -> N15;
  N15 -> N18;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N22 -> N23;
  N23 -> N26;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N30 -> N31;
  N31 -> N34;
  N32 -> N33;
  N33 -> N34;
  N34 -> N35;
  N35 -> N36;
  N36 -> N37;
  N38 -> N39;
  N39 -> N42;
  N40 -> N41;
  N41 -> N42;
  N42 -> N43;
  N43 -> N44;
  N44 -> N45;
  N46 -> N47;
  N47 -> N50;
  N48 -> N49;
  N49 -> N50;
  N50 -> N51;
  N51 -> N52;
  N52 -> N53;
  N54 -> N55;
  N55 -> N58;
  N56 -> N57;
  N57 -> N58;
  N58 -> N59;
  N59 -> N60;
  N60 -> N61;
  N62 -> N63;
  N63 -> N66;
  N64 -> N65;
  N65 -> N66;
  N66 -> N67;
  N67 -> N68;
  N68 -> N69;
}
"""


class LogisticRegressionTest(unittest.TestCase):
    def test_logistic_regression_inference(self) -> None:
        self.maxDiff = None
        bmg = BMGInference()
        samples = bmg.infer(queries, observations, 1000)
        b0 = samples[beta(0)].mean()
        b1 = samples[beta(1)].mean()
        b2 = samples[beta(2)].mean()

        slope_ob = -b1 / b2
        int_ob = -b0 / b2
        slope_ex = 0.64  # Should be 0.67
        int_ex = 0.16  # Should be -0.33; reasonable guess given thin data

        self.assertAlmostEqual(first=slope_ob, second=slope_ex, delta=0.05)
        self.assertAlmostEqual(first=int_ob, second=int_ex, delta=0.05)

    def test_logistic_regression_to_dot(self) -> None:
        self.maxDiff = None
        bmg = BMGInference()
        observed = bmg.to_dot(queries, observations)
        self.assertEqual(expected_dot.strip(), observed.strip())
