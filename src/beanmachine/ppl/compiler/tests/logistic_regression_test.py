# Copyright (c) Facebook, Inc. and its affiliates.
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
  N08[label="+"];
  N09[label=5.6988];
  N10[label="*"];
  N11[label="+"];
  N12[label="Bernoulli(logits)"];
  N13[label=Sample];
  N14[label="Observation False"];
  N15[label=-6.2928];
  N16[label="*"];
  N17[label="+"];
  N18[label=1.1692];
  N19[label="*"];
  N20[label="+"];
  N21[label="Bernoulli(logits)"];
  N22[label=Sample];
  N23[label="Observation False"];
  N24[label=1.6583];
  N25[label="*"];
  N26[label="+"];
  N27[label=-4.7142];
  N28[label="*"];
  N29[label="+"];
  N30[label="Bernoulli(logits)"];
  N31[label=Sample];
  N32[label="Observation True"];
  N33[label=-7.7588];
  N34[label="*"];
  N35[label="+"];
  N36[label=7.9859];
  N37[label="*"];
  N38[label="+"];
  N39[label="Bernoulli(logits)"];
  N40[label=Sample];
  N41[label="Observation False"];
  N42[label=-1.2421];
  N43[label="*"];
  N44[label="+"];
  N45[label=5.4628];
  N46[label="*"];
  N47[label="+"];
  N48[label="Bernoulli(logits)"];
  N49[label=Sample];
  N50[label="Observation False"];
  N51[label=6.4529];
  N52[label="*"];
  N53[label="+"];
  N54[label=2.3994];
  N55[label="*"];
  N56[label="+"];
  N57[label="Bernoulli(logits)"];
  N58[label=Sample];
  N59[label="Observation True"];
  N60[label=-4.9269];
  N61[label="*"];
  N62[label="+"];
  N63[label=7.8679];
  N64[label="*"];
  N65[label="+"];
  N66[label="Bernoulli(logits)"];
  N67[label=Sample];
  N68[label="Observation False"];
  N69[label=4.213];
  N70[label="*"];
  N71[label="+"];
  N72[label=2.6175];
  N73[label="*"];
  N74[label="+"];
  N75[label="Bernoulli(logits)"];
  N76[label=Sample];
  N77[label="Observation True"];
  N78[label=Query];
  N79[label=Query];
  N80[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N03 -> N08;
  N03 -> N17;
  N03 -> N26;
  N03 -> N35;
  N03 -> N44;
  N03 -> N53;
  N03 -> N62;
  N03 -> N71;
  N03 -> N78;
  N04 -> N07;
  N04 -> N16;
  N04 -> N25;
  N04 -> N34;
  N04 -> N43;
  N04 -> N52;
  N04 -> N61;
  N04 -> N70;
  N04 -> N79;
  N05 -> N10;
  N05 -> N19;
  N05 -> N28;
  N05 -> N37;
  N05 -> N46;
  N05 -> N55;
  N05 -> N64;
  N05 -> N73;
  N05 -> N80;
  N06 -> N07;
  N07 -> N08;
  N08 -> N11;
  N09 -> N10;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N15 -> N16;
  N16 -> N17;
  N17 -> N20;
  N18 -> N19;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N24 -> N25;
  N25 -> N26;
  N26 -> N29;
  N27 -> N28;
  N28 -> N29;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
  N33 -> N34;
  N34 -> N35;
  N35 -> N38;
  N36 -> N37;
  N37 -> N38;
  N38 -> N39;
  N39 -> N40;
  N40 -> N41;
  N42 -> N43;
  N43 -> N44;
  N44 -> N47;
  N45 -> N46;
  N46 -> N47;
  N47 -> N48;
  N48 -> N49;
  N49 -> N50;
  N51 -> N52;
  N52 -> N53;
  N53 -> N56;
  N54 -> N55;
  N55 -> N56;
  N56 -> N57;
  N57 -> N58;
  N58 -> N59;
  N60 -> N61;
  N61 -> N62;
  N62 -> N65;
  N63 -> N64;
  N64 -> N65;
  N65 -> N66;
  N66 -> N67;
  N67 -> N68;
  N69 -> N70;
  N70 -> N71;
  N71 -> N74;
  N72 -> N73;
  N73 -> N74;
  N74 -> N75;
  N75 -> N76;
  N76 -> N77;
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
        self.assertEqual(observed.strip(), expected_dot.strip())
