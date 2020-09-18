# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test of realistic logistic regression model"""
import unittest

from beanmachine.ppl.compiler.bm_to_bmg import to_bmg, to_dot


def tidy(s: str) -> str:
    return "\n".join(c.strip() for c in s.strip().split("\n")).strip()


source = """
import beanmachine.ppl as bm
import torch
from torch.distributions import Bernoulli, Normal

# We have N points with K coordinates each classified into one
# of two categories.

# Points are generated so that posteriors should be
# centered on beta(0) around -1.0, beta(1) around 2.0,
# beta(2) around -3.0

N = 8
K = 2
X = [
    [ 1.0000,  7.6483,  5.6988],
    [ 1.0000, -6.2928,  1.1692],
    [ 1.0000,  1.6583, -4.7142],
    [ 1.0000, -7.7588,  7.9859],
    [ 1.0000, -1.2421,  5.4628],
    [ 1.0000,  6.4529,  2.3994],
    [ 1.0000, -4.9269,  7.8679],
    [ 1.0000,  4.2130,  2.6175]]

# Classifications of those N points into two buckets:

Y = [0., 0., 1., 0., 0., 1., 0., 1.]

@bm.random_variable
def beta(k): # k is 0 to K
    return Normal(0.0, 1.0)

@bm.random_variable
def y(n): # n is 0 to N-1
    mu = X[n][0] * beta(0) + X[n][1] * beta(1) + X[n][2] * beta(2)
    return Bernoulli(logits=mu)

y(0)
y(1)
y(2)
y(3)
y(4)
y(5)
y(6)
y(7)
"""

expected_bmg = """
Node 0 type 1 parents [ ] children [ 2 ] real 0
Node 1 type 1 parents [ ] children [ 2 ] positive real 1
Node 2 type 2 parents [ 0 1 ] children [ 3 4 5 ] unknown
Node 3 type 3 parents [ 2 ] children [ 7 ] real 0
Node 4 type 3 parents [ 2 ] children [ 9 17 25 33 41 49 57 65 ] real 0
Node 5 type 3 parents [ 2 ] children [ 12 20 28 36 44 52 60 68 ] real 0
Node 6 type 1 parents [ ] children [ 7 ] real 1
Node 7 type 3 parents [ 6 3 ] children [ 10 18 26 34 42 50 58 66 ] real 0
Node 8 type 1 parents [ ] children [ 9 ] real 7.6483
Node 9 type 3 parents [ 8 4 ] children [ 10 ] real 0
Node 10 type 3 parents [ 7 9 ] children [ 13 ] real 0
Node 11 type 1 parents [ ] children [ 12 ] real 5.6988
Node 12 type 3 parents [ 11 5 ] children [ 13 ] real 0
Node 13 type 3 parents [ 10 12 ] children [ 14 ] real 0
Node 14 type 2 parents [ 13 ] children [ 15 ] unknown
Node 15 type 3 parents [ 14 ] children [ ] boolean 0
Node 16 type 1 parents [ ] children [ 17 ] real -6.2928
Node 17 type 3 parents [ 16 4 ] children [ 18 ] real 0
Node 18 type 3 parents [ 7 17 ] children [ 21 ] real 0
Node 19 type 1 parents [ ] children [ 20 ] real 1.1692
Node 20 type 3 parents [ 19 5 ] children [ 21 ] real 0
Node 21 type 3 parents [ 18 20 ] children [ 22 ] real 0
Node 22 type 2 parents [ 21 ] children [ 23 ] unknown
Node 23 type 3 parents [ 22 ] children [ ] boolean 0
Node 24 type 1 parents [ ] children [ 25 ] real 1.6583
Node 25 type 3 parents [ 24 4 ] children [ 26 ] real 0
Node 26 type 3 parents [ 7 25 ] children [ 29 ] real 0
Node 27 type 1 parents [ ] children [ 28 ] real -4.7142
Node 28 type 3 parents [ 27 5 ] children [ 29 ] real 0
Node 29 type 3 parents [ 26 28 ] children [ 30 ] real 0
Node 30 type 2 parents [ 29 ] children [ 31 ] unknown
Node 31 type 3 parents [ 30 ] children [ ] boolean 0
Node 32 type 1 parents [ ] children [ 33 ] real -7.7588
Node 33 type 3 parents [ 32 4 ] children [ 34 ] real 0
Node 34 type 3 parents [ 7 33 ] children [ 37 ] real 0
Node 35 type 1 parents [ ] children [ 36 ] real 7.9859
Node 36 type 3 parents [ 35 5 ] children [ 37 ] real 0
Node 37 type 3 parents [ 34 36 ] children [ 38 ] real 0
Node 38 type 2 parents [ 37 ] children [ 39 ] unknown
Node 39 type 3 parents [ 38 ] children [ ] boolean 0
Node 40 type 1 parents [ ] children [ 41 ] real -1.2421
Node 41 type 3 parents [ 40 4 ] children [ 42 ] real 0
Node 42 type 3 parents [ 7 41 ] children [ 45 ] real 0
Node 43 type 1 parents [ ] children [ 44 ] real 5.4628
Node 44 type 3 parents [ 43 5 ] children [ 45 ] real 0
Node 45 type 3 parents [ 42 44 ] children [ 46 ] real 0
Node 46 type 2 parents [ 45 ] children [ 47 ] unknown
Node 47 type 3 parents [ 46 ] children [ ] boolean 0
Node 48 type 1 parents [ ] children [ 49 ] real 6.4529
Node 49 type 3 parents [ 48 4 ] children [ 50 ] real 0
Node 50 type 3 parents [ 7 49 ] children [ 53 ] real 0
Node 51 type 1 parents [ ] children [ 52 ] real 2.3994
Node 52 type 3 parents [ 51 5 ] children [ 53 ] real 0
Node 53 type 3 parents [ 50 52 ] children [ 54 ] real 0
Node 54 type 2 parents [ 53 ] children [ 55 ] unknown
Node 55 type 3 parents [ 54 ] children [ ] boolean 0
Node 56 type 1 parents [ ] children [ 57 ] real -4.9269
Node 57 type 3 parents [ 56 4 ] children [ 58 ] real 0
Node 58 type 3 parents [ 7 57 ] children [ 61 ] real 0
Node 59 type 1 parents [ ] children [ 60 ] real 7.8679
Node 60 type 3 parents [ 59 5 ] children [ 61 ] real 0
Node 61 type 3 parents [ 58 60 ] children [ 62 ] real 0
Node 62 type 2 parents [ 61 ] children [ 63 ] unknown
Node 63 type 3 parents [ 62 ] children [ ] boolean 0
Node 64 type 1 parents [ ] children [ 65 ] real 4.213
Node 65 type 3 parents [ 64 4 ] children [ 66 ] real 0
Node 66 type 3 parents [ 7 65 ] children [ 69 ] real 0
Node 67 type 1 parents [ ] children [ 68 ] real 2.6175
Node 68 type 3 parents [ 67 5 ] children [ 69 ] real 0
Node 69 type 3 parents [ 66 68 ] children [ 70 ] real 0
Node 70 type 2 parents [ 69 ] children [ 71 ] unknown
Node 71 type 3 parents [ 70 ] children [ ] boolean 0
"""


expected_dot = """
digraph "graph" {
  N00[label="0.0:R"];
  N01[label="1.0:R+"];
  N02[label="Normal:R"];
  N03[label="Sample:R"];
  N04[label="Sample:R"];
  N05[label="Sample:R"];
  N06[label="1.0:R"];
  N07[label="*:R"];
  N08[label="7.6483:R"];
  N09[label="*:R"];
  N10[label="+:R"];
  N11[label="5.6988:R"];
  N12[label="*:R"];
  N13[label="+:R"];
  N14[label="Bernoulli(logits):B"];
  N15[label="Sample:B"];
  N16[label="-6.2928:R"];
  N17[label="*:R"];
  N18[label="+:R"];
  N19[label="1.1692:R"];
  N20[label="*:R"];
  N21[label="+:R"];
  N22[label="Bernoulli(logits):B"];
  N23[label="Sample:B"];
  N24[label="1.6583:R"];
  N25[label="*:R"];
  N26[label="+:R"];
  N27[label="-4.7142:R"];
  N28[label="*:R"];
  N29[label="+:R"];
  N30[label="Bernoulli(logits):B"];
  N31[label="Sample:B"];
  N32[label="-7.7588:R"];
  N33[label="*:R"];
  N34[label="+:R"];
  N35[label="7.9859:R"];
  N36[label="*:R"];
  N37[label="+:R"];
  N38[label="Bernoulli(logits):B"];
  N39[label="Sample:B"];
  N40[label="-1.2421:R"];
  N41[label="*:R"];
  N42[label="+:R"];
  N43[label="5.4628:R"];
  N44[label="*:R"];
  N45[label="+:R"];
  N46[label="Bernoulli(logits):B"];
  N47[label="Sample:B"];
  N48[label="6.4529:R"];
  N49[label="*:R"];
  N50[label="+:R"];
  N51[label="2.3994:R"];
  N52[label="*:R"];
  N53[label="+:R"];
  N54[label="Bernoulli(logits):B"];
  N55[label="Sample:B"];
  N56[label="-4.9269:R"];
  N57[label="*:R"];
  N58[label="+:R"];
  N59[label="7.8679:R"];
  N60[label="*:R"];
  N61[label="+:R"];
  N62[label="Bernoulli(logits):B"];
  N63[label="Sample:B"];
  N64[label="4.213:R"];
  N65[label="*:R"];
  N66[label="+:R"];
  N67[label="2.6175:R"];
  N68[label="*:R"];
  N69[label="+:R"];
  N70[label="Bernoulli(logits):B"];
  N71[label="Sample:B"];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N04[label=operand];
  N02 -> N05[label=operand];
  N03 -> N07[label=right];
  N04 -> N09[label=right];
  N04 -> N17[label=right];
  N04 -> N25[label=right];
  N04 -> N33[label=right];
  N04 -> N41[label=right];
  N04 -> N49[label=right];
  N04 -> N57[label=right];
  N04 -> N65[label=right];
  N05 -> N12[label=right];
  N05 -> N20[label=right];
  N05 -> N28[label=right];
  N05 -> N36[label=right];
  N05 -> N44[label=right];
  N05 -> N52[label=right];
  N05 -> N60[label=right];
  N05 -> N68[label=right];
  N06 -> N07[label=left];
  N07 -> N10[label=left];
  N07 -> N18[label=left];
  N07 -> N26[label=left];
  N07 -> N34[label=left];
  N07 -> N42[label=left];
  N07 -> N50[label=left];
  N07 -> N58[label=left];
  N07 -> N66[label=left];
  N08 -> N09[label=left];
  N09 -> N10[label=right];
  N10 -> N13[label=left];
  N11 -> N12[label=left];
  N12 -> N13[label=right];
  N13 -> N14[label=probability];
  N14 -> N15[label=operand];
  N16 -> N17[label=left];
  N17 -> N18[label=right];
  N18 -> N21[label=left];
  N19 -> N20[label=left];
  N20 -> N21[label=right];
  N21 -> N22[label=probability];
  N22 -> N23[label=operand];
  N24 -> N25[label=left];
  N25 -> N26[label=right];
  N26 -> N29[label=left];
  N27 -> N28[label=left];
  N28 -> N29[label=right];
  N29 -> N30[label=probability];
  N30 -> N31[label=operand];
  N32 -> N33[label=left];
  N33 -> N34[label=right];
  N34 -> N37[label=left];
  N35 -> N36[label=left];
  N36 -> N37[label=right];
  N37 -> N38[label=probability];
  N38 -> N39[label=operand];
  N40 -> N41[label=left];
  N41 -> N42[label=right];
  N42 -> N45[label=left];
  N43 -> N44[label=left];
  N44 -> N45[label=right];
  N45 -> N46[label=probability];
  N46 -> N47[label=operand];
  N48 -> N49[label=left];
  N49 -> N50[label=right];
  N50 -> N53[label=left];
  N51 -> N52[label=left];
  N52 -> N53[label=right];
  N53 -> N54[label=probability];
  N54 -> N55[label=operand];
  N56 -> N57[label=left];
  N57 -> N58[label=right];
  N58 -> N61[label=left];
  N59 -> N60[label=left];
  N60 -> N61[label=right];
  N61 -> N62[label=probability];
  N62 -> N63[label=operand];
  N64 -> N65[label=left];
  N65 -> N66[label=right];
  N66 -> N69[label=left];
  N67 -> N68[label=left];
  N68 -> N69[label=right];
  N69 -> N70[label=probability];
  N70 -> N71[label=operand];
}
"""


class LogisticRegressionTest(unittest.TestCase):
    def test_to_bmg(self) -> None:
        """test_to_bmg from logistic_regression_test.py"""
        self.maxDiff = None
        observed = to_bmg(source).to_string()
        self.assertEqual(tidy(observed), tidy(expected_bmg))

    def test_to_dot(self) -> None:
        """test_to_dot from logistic_regression_test.py"""
        self.maxDiff = None
        observed = to_dot(
            source=source,
            graph_types=True,
            inf_types=False,
            edge_requirements=False,
            point_at_input=True,
            after_transform=True,
        )
        self.assertEqual(observed.strip(), expected_dot.strip())
