# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


@bm.random_variable
def beta(n):
    return Beta(2.0, 2.0)


@bm.random_variable
def flip_beta():
    return Bernoulli(tensor([beta(0), beta(1)]))


@bm.random_variable
def flip_const():
    return Bernoulli(tensor([0.25, 0.75]))


class FixVectorizedModelsTest(unittest.TestCase):
    def test_fix_vectorized_models_1(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [flip_beta(), flip_const()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=Tensor];
  N05[label=Bernoulli];
  N06[label=Sample];
  N07[label=Query];
  N08[label="[0.25,0.75]"];
  N09[label=Bernoulli];
  N10[label=Sample];
  N11[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N04;
  N03 -> N04;
  N04 -> N05;
  N05 -> N06;
  N06 -> N07;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=2.0];
  N01[label=Beta];
  N02[label=Sample];
  N03[label=Sample];
  N04[label=2];
  N05[label=1];
  N06[label=Bernoulli];
  N07[label=Sample];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label=ToMatrix];
  N11[label=Query];
  N12[label=0.25];
  N13[label=Bernoulli];
  N14[label=Sample];
  N15[label=0.75];
  N16[label=Bernoulli];
  N17[label=Sample];
  N18[label=ToMatrix];
  N19[label=Query];
  N00 -> N01;
  N00 -> N01;
  N01 -> N02;
  N01 -> N03;
  N02 -> N06;
  N03 -> N08;
  N04 -> N10;
  N04 -> N18;
  N05 -> N10;
  N05 -> N18;
  N06 -> N07;
  N07 -> N10;
  N08 -> N09;
  N09 -> N10;
  N10 -> N11;
  N12 -> N13;
  N13 -> N14;
  N14 -> N18;
  N15 -> N16;
  N16 -> N17;
  N17 -> N18;
  N18 -> N19;
}
"""
        self.assertEqual(expected.strip(), observed.strip())
