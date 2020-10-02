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
Node 3 type 3 parents [ 2 ] children [ 8 16 24 32 40 48 56 64 ] real 1e-10
Node 4 type 3 parents [ 2 ] children [ 7 15 23 31 39 47 55 63 ] real 1e-10
Node 5 type 3 parents [ 2 ] children [ 10 18 26 34 42 50 58 66 ] real 1e-10
Node 6 type 1 parents [ ] children [ 7 ] real 7.6483
Node 7 type 3 parents [ 6 4 ] children [ 8 ] real 1e-10
Node 8 type 3 parents [ 3 7 ] children [ 11 ] real 1e-10
Node 9 type 1 parents [ ] children [ 10 ] real 5.6988
Node 10 type 3 parents [ 9 5 ] children [ 11 ] real 1e-10
Node 11 type 3 parents [ 8 10 ] children [ 12 ] real 1e-10
Node 12 type 2 parents [ 11 ] children [ 13 ] unknown
Node 13 type 3 parents [ 12 ] children [ ] boolean 0
Node 14 type 1 parents [ ] children [ 15 ] real -6.2928
Node 15 type 3 parents [ 14 4 ] children [ 16 ] real 1e-10
Node 16 type 3 parents [ 3 15 ] children [ 19 ] real 1e-10
Node 17 type 1 parents [ ] children [ 18 ] real 1.1692
Node 18 type 3 parents [ 17 5 ] children [ 19 ] real 1e-10
Node 19 type 3 parents [ 16 18 ] children [ 20 ] real 1e-10
Node 20 type 2 parents [ 19 ] children [ 21 ] unknown
Node 21 type 3 parents [ 20 ] children [ ] boolean 0
Node 22 type 1 parents [ ] children [ 23 ] real 1.6583
Node 23 type 3 parents [ 22 4 ] children [ 24 ] real 1e-10
Node 24 type 3 parents [ 3 23 ] children [ 27 ] real 1e-10
Node 25 type 1 parents [ ] children [ 26 ] real -4.7142
Node 26 type 3 parents [ 25 5 ] children [ 27 ] real 1e-10
Node 27 type 3 parents [ 24 26 ] children [ 28 ] real 1e-10
Node 28 type 2 parents [ 27 ] children [ 29 ] unknown
Node 29 type 3 parents [ 28 ] children [ ] boolean 0
Node 30 type 1 parents [ ] children [ 31 ] real -7.7588
Node 31 type 3 parents [ 30 4 ] children [ 32 ] real 1e-10
Node 32 type 3 parents [ 3 31 ] children [ 35 ] real 1e-10
Node 33 type 1 parents [ ] children [ 34 ] real 7.9859
Node 34 type 3 parents [ 33 5 ] children [ 35 ] real 1e-10
Node 35 type 3 parents [ 32 34 ] children [ 36 ] real 1e-10
Node 36 type 2 parents [ 35 ] children [ 37 ] unknown
Node 37 type 3 parents [ 36 ] children [ ] boolean 0
Node 38 type 1 parents [ ] children [ 39 ] real -1.2421
Node 39 type 3 parents [ 38 4 ] children [ 40 ] real 1e-10
Node 40 type 3 parents [ 3 39 ] children [ 43 ] real 1e-10
Node 41 type 1 parents [ ] children [ 42 ] real 5.4628
Node 42 type 3 parents [ 41 5 ] children [ 43 ] real 1e-10
Node 43 type 3 parents [ 40 42 ] children [ 44 ] real 1e-10
Node 44 type 2 parents [ 43 ] children [ 45 ] unknown
Node 45 type 3 parents [ 44 ] children [ ] boolean 0
Node 46 type 1 parents [ ] children [ 47 ] real 6.4529
Node 47 type 3 parents [ 46 4 ] children [ 48 ] real 1e-10
Node 48 type 3 parents [ 3 47 ] children [ 51 ] real 1e-10
Node 49 type 1 parents [ ] children [ 50 ] real 2.3994
Node 50 type 3 parents [ 49 5 ] children [ 51 ] real 1e-10
Node 51 type 3 parents [ 48 50 ] children [ 52 ] real 1e-10
Node 52 type 2 parents [ 51 ] children [ 53 ] unknown
Node 53 type 3 parents [ 52 ] children [ ] boolean 0
Node 54 type 1 parents [ ] children [ 55 ] real -4.9269
Node 55 type 3 parents [ 54 4 ] children [ 56 ] real 1e-10
Node 56 type 3 parents [ 3 55 ] children [ 59 ] real 1e-10
Node 57 type 1 parents [ ] children [ 58 ] real 7.8679
Node 58 type 3 parents [ 57 5 ] children [ 59 ] real 1e-10
Node 59 type 3 parents [ 56 58 ] children [ 60 ] real 1e-10
Node 60 type 2 parents [ 59 ] children [ 61 ] unknown
Node 61 type 3 parents [ 60 ] children [ ] boolean 0
Node 62 type 1 parents [ ] children [ 63 ] real 4.213
Node 63 type 3 parents [ 62 4 ] children [ 64 ] real 1e-10
Node 64 type 3 parents [ 3 63 ] children [ 67 ] real 1e-10
Node 65 type 1 parents [ ] children [ 66 ] real 2.6175
Node 66 type 3 parents [ 65 5 ] children [ 67 ] real 1e-10
Node 67 type 3 parents [ 64 66 ] children [ 68 ] real 1e-10
Node 68 type 2 parents [ 67 ] children [ 69 ] unknown
Node 69 type 3 parents [ 68 ] children [ ] boolean 0
"""


expected_dot = """
digraph "graph" {
  N00[label="0.0:R"];
  N01[label="1.0:R+"];
  N02[label="Normal:R"];
  N03[label="Sample:R"];
  N04[label="Sample:R"];
  N05[label="Sample:R"];
  N06[label="7.6483:R"];
  N07[label="*:R"];
  N08[label="+:R"];
  N09[label="5.6988:R"];
  N10[label="*:R"];
  N11[label="+:R"];
  N12[label="Bernoulli(logits):B"];
  N13[label="Sample:B"];
  N14[label="-6.2928:R"];
  N15[label="*:R"];
  N16[label="+:R"];
  N17[label="1.1692:R"];
  N18[label="*:R"];
  N19[label="+:R"];
  N20[label="Bernoulli(logits):B"];
  N21[label="Sample:B"];
  N22[label="1.6583:R"];
  N23[label="*:R"];
  N24[label="+:R"];
  N25[label="-4.7142:R"];
  N26[label="*:R"];
  N27[label="+:R"];
  N28[label="Bernoulli(logits):B"];
  N29[label="Sample:B"];
  N30[label="-7.7588:R"];
  N31[label="*:R"];
  N32[label="+:R"];
  N33[label="7.9859:R"];
  N34[label="*:R"];
  N35[label="+:R"];
  N36[label="Bernoulli(logits):B"];
  N37[label="Sample:B"];
  N38[label="-1.2421:R"];
  N39[label="*:R"];
  N40[label="+:R"];
  N41[label="5.4628:R"];
  N42[label="*:R"];
  N43[label="+:R"];
  N44[label="Bernoulli(logits):B"];
  N45[label="Sample:B"];
  N46[label="6.4529:R"];
  N47[label="*:R"];
  N48[label="+:R"];
  N49[label="2.3994:R"];
  N50[label="*:R"];
  N51[label="+:R"];
  N52[label="Bernoulli(logits):B"];
  N53[label="Sample:B"];
  N54[label="-4.9269:R"];
  N55[label="*:R"];
  N56[label="+:R"];
  N57[label="7.8679:R"];
  N58[label="*:R"];
  N59[label="+:R"];
  N60[label="Bernoulli(logits):B"];
  N61[label="Sample:B"];
  N62[label="4.213:R"];
  N63[label="*:R"];
  N64[label="+:R"];
  N65[label="2.6175:R"];
  N66[label="*:R"];
  N67[label="+:R"];
  N68[label="Bernoulli(logits):B"];
  N69[label="Sample:B"];
  N00 -> N02[label=mu];
  N01 -> N02[label=sigma];
  N02 -> N03[label=operand];
  N02 -> N04[label=operand];
  N02 -> N05[label=operand];
  N03 -> N08[label=left];
  N03 -> N16[label=left];
  N03 -> N24[label=left];
  N03 -> N32[label=left];
  N03 -> N40[label=left];
  N03 -> N48[label=left];
  N03 -> N56[label=left];
  N03 -> N64[label=left];
  N04 -> N07[label=right];
  N04 -> N15[label=right];
  N04 -> N23[label=right];
  N04 -> N31[label=right];
  N04 -> N39[label=right];
  N04 -> N47[label=right];
  N04 -> N55[label=right];
  N04 -> N63[label=right];
  N05 -> N10[label=right];
  N05 -> N18[label=right];
  N05 -> N26[label=right];
  N05 -> N34[label=right];
  N05 -> N42[label=right];
  N05 -> N50[label=right];
  N05 -> N58[label=right];
  N05 -> N66[label=right];
  N06 -> N07[label=left];
  N07 -> N08[label=right];
  N08 -> N11[label=left];
  N09 -> N10[label=left];
  N10 -> N11[label=right];
  N11 -> N12[label=probability];
  N12 -> N13[label=operand];
  N14 -> N15[label=left];
  N15 -> N16[label=right];
  N16 -> N19[label=left];
  N17 -> N18[label=left];
  N18 -> N19[label=right];
  N19 -> N20[label=probability];
  N20 -> N21[label=operand];
  N22 -> N23[label=left];
  N23 -> N24[label=right];
  N24 -> N27[label=left];
  N25 -> N26[label=left];
  N26 -> N27[label=right];
  N27 -> N28[label=probability];
  N28 -> N29[label=operand];
  N30 -> N31[label=left];
  N31 -> N32[label=right];
  N32 -> N35[label=left];
  N33 -> N34[label=left];
  N34 -> N35[label=right];
  N35 -> N36[label=probability];
  N36 -> N37[label=operand];
  N38 -> N39[label=left];
  N39 -> N40[label=right];
  N40 -> N43[label=left];
  N41 -> N42[label=left];
  N42 -> N43[label=right];
  N43 -> N44[label=probability];
  N44 -> N45[label=operand];
  N46 -> N47[label=left];
  N47 -> N48[label=right];
  N48 -> N51[label=left];
  N49 -> N50[label=left];
  N50 -> N51[label=right];
  N51 -> N52[label=probability];
  N52 -> N53[label=operand];
  N54 -> N55[label=left];
  N55 -> N56[label=right];
  N56 -> N59[label=left];
  N57 -> N58[label=left];
  N58 -> N59[label=right];
  N59 -> N60[label=probability];
  N60 -> N61[label=operand];
  N62 -> N63[label=left];
  N63 -> N64[label=right];
  N64 -> N67[label=left];
  N65 -> N66[label=left];
  N66 -> N67[label=right];
  N67 -> N68[label=probability];
  N68 -> N69[label=operand];
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
