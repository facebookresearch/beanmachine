# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


# This is a very simplified version of a CLARA model; this is the sort of model
# that we want to apply our workaround of removing observations on.


@bm.random_variable
def sensitivity(labeler):
    return Beta(1, 1)


@bm.random_variable
def specificity(labeler):
    return Beta(2, 2)


@bm.random_variable
def prevalence():
    return Beta(0.5, 0.5)


@bm.random_variable
def observation(x):
    bob = 0
    sue = 1
    pos_sum = prevalence().log() + sensitivity(bob).log() + sensitivity(sue).log()
    neg_sum = (
        (1 - prevalence()).log()
        + (1 - specificity(bob)).log()
        + (1 - specificity(sue)).log()
    )
    log_prob = (pos_sum.exp() + neg_sum.exp()).log()
    return Bernoulli(log_prob.exp())


class FixObserveTrueTest(unittest.TestCase):
    def test_fix_observe_true(self) -> None:
        self.maxDiff = None
        observations = {observation(0): tensor(1.0), observation(1): tensor(1.0)}
        queries = []

        bmg = BMGInference()
        observed = bmg.to_dot(queries, observations)

        # Here's the model as it would be handed off to BMG normally.

        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=1.0];
  N04[label=Beta];
  N05[label=Sample];
  N06[label=Sample];
  N07[label=2.0];
  N08[label=Beta];
  N09[label=Sample];
  N10[label=Sample];
  N11[label=Log];
  N12[label=Log];
  N13[label=Log];
  N14[label="+"];
  N15[label=complement];
  N16[label=Log];
  N17[label=complement];
  N18[label=Log];
  N19[label=complement];
  N20[label=Log];
  N21[label="+"];
  N22[label=LogSumExp];
  N23[label=Exp];
  N24[label=ToProb];
  N25[label=Bernoulli];
  N26[label=Sample];
  N27[label="Observation True"];
  N28[label=Sample];
  N29[label="Observation True"];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N11;
  N02 -> N15;
  N03 -> N04;
  N03 -> N04;
  N04 -> N05;
  N04 -> N06;
  N05 -> N12;
  N06 -> N13;
  N07 -> N08;
  N07 -> N08;
  N08 -> N09;
  N08 -> N10;
  N09 -> N17;
  N10 -> N19;
  N11 -> N14;
  N12 -> N14;
  N13 -> N14;
  N14 -> N22;
  N15 -> N16;
  N16 -> N21;
  N17 -> N18;
  N18 -> N21;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N25 -> N28;
  N26 -> N27;
  N28 -> N29;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # Now let's force an additional rewriting pass. Note that there must
        # be as many factor nodes as we removed observations; factor nodes
        # are not deduplicated.

        bmg = BMGInference()
        bmg._fix_observe_true = True
        observed = bmg.to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=0.5];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=1.0];
  N04[label=Beta];
  N05[label=Sample];
  N06[label=Sample];
  N07[label=2.0];
  N08[label=Beta];
  N09[label=Sample];
  N10[label=Sample];
  N11[label=Log];
  N12[label=Log];
  N13[label=Log];
  N14[label="+"];
  N15[label=complement];
  N16[label=Log];
  N17[label=complement];
  N18[label=Log];
  N19[label=complement];
  N20[label=Log];
  N21[label="+"];
  N22[label=LogSumExp];
  N23[label=ExpProduct];
  N24[label=ExpProduct];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N11;
  N02 -> N15;
  N03 -> N04;
  N03 -> N04;
  N04 -> N05;
  N04 -> N06;
  N05 -> N12;
  N06 -> N13;
  N07 -> N08;
  N07 -> N08;
  N08 -> N09;
  N08 -> N10;
  N09 -> N17;
  N10 -> N19;
  N11 -> N14;
  N12 -> N14;
  N13 -> N14;
  N14 -> N22;
  N15 -> N16;
  N16 -> N21;
  N17 -> N18;
  N18 -> N21;
  N19 -> N20;
  N20 -> N21;
  N21 -> N22;
  N22 -> N23;
  N22 -> N24;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
