# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Normal


@bm.random_variable
def n(n):
    return Normal(0, 1)


@bm.random_variable
def n12():
    return Normal(tensor([n(3), n(4)]), 1.0)


@bm.random_variable
def n21():
    return Normal(tensor([[n(1)], [n(2)]]), 1.0)


@bm.functional
def broadcast_add():
    return n12() + n21()


class BroadcastTest(unittest.TestCase):
    def test_broadcast_add(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [broadcast_add()]

        observed = BMGInference().to_dot(queries, observations, after_transform=False)

        # The model before the rewrite:

        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Tensor];
  N06[label=1.0];
  N07[label=Normal];
  N08[label=Sample];
  N09[label=Sample];
  N10[label=Sample];
  N11[label=Tensor];
  N12[label=Normal];
  N13[label=Sample];
  N14[label="+"];
  N15[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N09;
  N02 -> N10;
  N03 -> N05;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N06 -> N12;
  N07 -> N08;
  N08 -> N14;
  N09 -> N11;
  N10 -> N11;
  N11 -> N12;
  N12 -> N13;
  N13 -> N14;
  N14 -> N15;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # After:

        observed = BMGInference().to_dot(queries, observations, after_transform=True)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Normal];
  N06[label=Sample];
  N07[label=Normal];
  N08[label=Sample];
  N09[label=Sample];
  N10[label=Sample];
  N11[label=Normal];
  N12[label=Sample];
  N13[label=Normal];
  N14[label=Sample];
  N15[label=2];
  N16[label=1];
  N17[label=ToMatrix];
  N18[label=ToMatrix];
  N19[label=MatrixAdd];
  N20[label=Query];
  N00 -> N02;
  N01 -> N02;
  N01 -> N05;
  N01 -> N07;
  N01 -> N11;
  N01 -> N13;
  N02 -> N03;
  N02 -> N04;
  N02 -> N09;
  N02 -> N10;
  N03 -> N05;
  N04 -> N07;
  N05 -> N06;
  N06 -> N17;
  N07 -> N08;
  N08 -> N17;
  N09 -> N11;
  N10 -> N13;
  N11 -> N12;
  N12 -> N18;
  N13 -> N14;
  N14 -> N18;
  N15 -> N17;
  N15 -> N18;
  N16 -> N17;
  N16 -> N18;
  N17 -> N19;
  N18 -> N19;
  N19 -> N20;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # BMG

        with self.assertRaises(ValueError):
            g, _ = BMGInference().to_graph(queries, observations)
        # observed = g.to_dot()
        # expected = ""
        # self.assertEqual(expected.strip(), observed.strip())
