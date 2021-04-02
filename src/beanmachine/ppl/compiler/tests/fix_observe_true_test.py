# Copyright (c) Facebook, Inc. and its affiliates.
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
def observation():
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
        observations = {observation(): tensor(1.0)}
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
  N15[label=Exp];
  N16[label=ToPosReal];
  N17[label=complement];
  N18[label=Log];
  N19[label=complement];
  N20[label=Log];
  N21[label=complement];
  N22[label=Log];
  N23[label="+"];
  N24[label=Exp];
  N25[label=ToPosReal];
  N26[label="+"];
  N27[label=Log];
  N28[label=Exp];
  N29[label=ToProb];
  N30[label=Bernoulli];
  N31[label=Sample];
  N32[label="Observation True"];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N11;
  N02 -> N17;
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
  N09 -> N19;
  N10 -> N21;
  N11 -> N14;
  N12 -> N14;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N26;
  N17 -> N18;
  N18 -> N23;
  N19 -> N20;
  N20 -> N23;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N28 -> N29;
  N29 -> N30;
  N30 -> N31;
  N31 -> N32;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # Now let's force an additional rewriting pass:
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
  N15[label=Exp];
  N16[label=ToPosReal];
  N17[label=complement];
  N18[label=Log];
  N19[label=complement];
  N20[label=Log];
  N21[label=complement];
  N22[label=Log];
  N23[label="+"];
  N24[label=Exp];
  N25[label=ToPosReal];
  N26[label="+"];
  N27[label=Log];
  N28[label=Exp];
  N29[label=ToProb];
  N30[label=Bernoulli];
  N31[label=Sample];
  N32[label=ExpProduct];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N02 -> N11;
  N02 -> N17;
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
  N09 -> N19;
  N10 -> N21;
  N11 -> N14;
  N12 -> N14;
  N13 -> N14;
  N14 -> N15;
  N15 -> N16;
  N16 -> N26;
  N17 -> N18;
  N18 -> N23;
  N19 -> N20;
  N20 -> N23;
  N21 -> N22;
  N22 -> N23;
  N23 -> N24;
  N24 -> N25;
  N25 -> N26;
  N26 -> N27;
  N27 -> N28;
  N27 -> N32;
  N28 -> N29;
  N29 -> N30;
  N30 -> N31;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
