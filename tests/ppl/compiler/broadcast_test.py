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


@bm.functional
def fill_add_1():
    return n12() + n(5)


@bm.functional
def fill_add_2():
    return n12() + 123


class BroadcastTest(unittest.TestCase):
    # TODO: Test broadcast multiplication as well.
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

        # We do not yet insert a broadcast node. Demonstrate here
        # that the compiler gives a reasonable error message.

        with self.assertRaises(ValueError) as ex:
            BMGInference().to_graph(queries, observations)

        expected = """
The left of a matrix add is required to be a 2 x 2 real matrix but is a 2 x 1 real matrix.
The right of a matrix add is required to be a 2 x 2 real matrix but is a 1 x 2 real matrix."""

        observed = str(ex.exception)
        self.assertEqual(expected.strip(), observed.strip())

    def test_fill_add_1(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [fill_add_1()]

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
  N10[label="+"];
  N11[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N09;
  N03 -> N05;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N10;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # The model after converting to BMG:

        g, _ = BMGInference().to_graph(queries, observations)
        observed = g.to_dot()

        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="~"];
  N5[label="Normal"];
  N6[label="~"];
  N7[label="Normal"];
  N8[label="~"];
  N9[label="~"];
  N10[label="2"];
  N11[label="1"];
  N12[label="ToMatrix"];
  N13[label="FillMatrix"];
  N14[label="MatrixAdd"];
  N0 -> N2;
  N1 -> N2;
  N1 -> N5;
  N1 -> N7;
  N2 -> N3;
  N2 -> N4;
  N2 -> N9;
  N3 -> N5;
  N4 -> N7;
  N5 -> N6;
  N6 -> N12;
  N7 -> N8;
  N8 -> N12;
  N9 -> N13;
  N10 -> N12;
  N10 -> N13;
  N11 -> N12;
  N11 -> N13;
  N12 -> N14;
  N13 -> N14;
  Q0[label="Query"];
  N14 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

    def test_fill_add_2(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [fill_add_2()]

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
  N09[label=123];
  N10[label="+"];
  N11[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N03 -> N05;
  N04 -> N05;
  N05 -> N07;
  N06 -> N07;
  N07 -> N08;
  N08 -> N10;
  N09 -> N10;
  N10 -> N11;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # The model after converting to BMG:

        # TODO: We could constant-fold the matrix fill here, though that might
        # be not actually an optimization if the matrix is large enough.

        g, _ = BMGInference().to_graph(queries, observations)
        observed = g.to_dot()

        expected = """
digraph "graph" {
  N0[label="0"];
  N1[label="1"];
  N2[label="Normal"];
  N3[label="~"];
  N4[label="~"];
  N5[label="Normal"];
  N6[label="~"];
  N7[label="Normal"];
  N8[label="~"];
  N9[label="2"];
  N10[label="1"];
  N11[label="ToMatrix"];
  N12[label="123"];
  N13[label="FillMatrix"];
  N14[label="MatrixAdd"];
  N0 -> N2;
  N1 -> N2;
  N1 -> N5;
  N1 -> N7;
  N2 -> N3;
  N2 -> N4;
  N3 -> N5;
  N4 -> N7;
  N5 -> N6;
  N6 -> N11;
  N7 -> N8;
  N8 -> N11;
  N9 -> N11;
  N9 -> N13;
  N10 -> N11;
  N10 -> N13;
  N11 -> N14;
  N12 -> N13;
  N13 -> N14;
  Q0[label="Query"];
  N14 -> Q0;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
