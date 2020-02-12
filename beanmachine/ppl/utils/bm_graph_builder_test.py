# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for bm_graph_builder.py"""
import unittest

from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from torch import tensor


class BMGraphBuilderTest(unittest.TestCase):
    def test_1(self) -> None:
        """Test 1"""

        bmg = BMGraphBuilder()
        half = bmg.add_real(0.5)
        two = bmg.add_real(2)
        tens = bmg.add_tensor(tensor([4.0, 5.0]))
        tr = bmg.add_boolean(True)
        flip = bmg.add_bernoulli(half)
        samp = bmg.add_sample(flip)
        real = bmg.add_to_real(samp)
        neg = bmg.add_negate(real)
        add = bmg.add_addition(two, neg)
        bmg.add_multiplication(tens, add)
        bmg.add_observation(flip, tr)

        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=0.5];
  N10[label=Observation];
  N1[label=2];
  N2[label="tensor([4., 5.])"];
  N3[label=True];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label=ToReal];
  N7[label="-"];
  N8[label="+"];
  N9[label="*"];
  N10 -> N3[label=value];
  N10 -> N4[label=operand];
  N4 -> N0[label=probability];
  N5 -> N4[label=operand];
  N6 -> N5[label=operand];
  N7 -> N6[label=operand];
  N8 -> N1[label=left];
  N8 -> N7[label=right];
  N9 -> N2[label=left];
  N9 -> N8[label=right];
}
"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())
