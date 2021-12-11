# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

# The item() member function should be treated as an identity by the compiler
# for the purposes of graph generation.

import beanmachine.ppl as bm
from beanmachine.ppl.inference.bmg_inference import BMGInference
from torch.distributions import Bernoulli, Beta


@bm.random_variable
def beta():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip():
    return Bernoulli(beta().item())


class ItemTest(unittest.TestCase):
    def test_item_member_function(self) -> None:

        self.maxDiff = None
        observed = BMGInference().to_dot([flip()], {}, after_transform=False)
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Item];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
  N5 -> N6;
}
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_dot([flip()], {}, after_transform=True)
        expected = """
digraph "graph" {
  N0[label=2.0];
  N1[label=Beta];
  N2[label=Sample];
  N3[label=Bernoulli];
  N4[label=Sample];
  N5[label=Query];
  N0 -> N1;
  N0 -> N1;
  N1 -> N2;
  N2 -> N3;
  N3 -> N4;
  N4 -> N5;
}
        """
        self.assertEqual(expected.strip(), observed.strip())
