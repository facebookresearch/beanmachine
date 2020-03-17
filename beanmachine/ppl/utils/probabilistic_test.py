# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for probabilistic.py"""
import unittest

from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from torch import tensor


class ProbabilisticTest(unittest.TestCase):
    """Tests for probabilistic.py"""

    def test_probabilistic_1(self) -> None:

        bmg: BMGraphBuilder = BMGraphBuilder()

        # x is 0 or 1, we use that to choose from two different distributions:
        #
        # @sample
        # def x():
        #   return Bernoulli(0.5)
        #
        # @sample
        # def sample_function_1(p):
        #   return Bernoulli((p + 0.5) * 0.5)
        #
        # z = sample_function_1(x())
        # We would lower that to something like:

        @probabilistic(bmg)
        @memoize
        def x():
            n1 = 0.5
            n2 = bmg.handle_bernoulli(n1)
            n3 = bmg.handle_sample(n2)
            return n3

        @probabilistic(bmg)
        @memoize
        def sample_function_1(p):
            n4 = 0.5
            n5 = bmg.handle_addition(p, n4)
            n6 = bmg.handle_multiplication(n5, n4)
            n7 = bmg.handle_bernoulli(n6)
            n8 = bmg.handle_sample(n7)
            return n8

        n9 = x()
        sample_function_1(n9)

        # Calling the function again should be a no-op; the nodes should be memoized

        sample_function_1(n9)

        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=0.5];
  N10[label=1.0];
  N11[label=map];
  N12[label=index];
  N1[label=Bernoulli];
  N2[label=Sample];
  N3[label=0.25];
  N4[label=Bernoulli];
  N5[label=Sample];
  N6[label=0.0];
  N7[label=0.75];
  N8[label=Bernoulli];
  N9[label=Sample];
  N1 -> N0[label=probability];
  N11 -> N10[label=2];
  N11 -> N5[label=1];
  N11 -> N6[label=0];
  N11 -> N9[label=3];
  N12 -> N11[label=left];
  N12 -> N2[label=right];
  N2 -> N1[label=operand];
  N4 -> N3[label=probability];
  N5 -> N4[label=operand];
  N8 -> N7[label=probability];
  N9 -> N8[label=operand];
}
        """
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())

    def test_probabilistic_2(self) -> None:

        bmg: BMGraphBuilder = BMGraphBuilder()

        # x is 0 or 1
        # y is 2 or 3
        # We want a Bernoulli(0.02) or (0.03) or (0.04)
        # @sample
        # def sample_function_2(x, y):
        #   return Bernoulli((x + y) * 0.01)
        # We would lower that to something like:
        @probabilistic(bmg)
        @memoize
        def sample_function_2(x, y):
            n1 = bmg.handle_addition(x, y)
            n2 = 0.01
            n3 = bmg.handle_multiplication(n1, n2)
            n4 = bmg.handle_bernoulli(n3)
            n5 = bmg.handle_sample(n4)
            return n5

        t2 = bmg.add_tensor(tensor(2.0))
        t5 = bmg.add_tensor(tensor(0.5))
        b = bmg.add_bernoulli(t5)
        s1 = bmg.add_sample(b)
        s2 = bmg.add_sample(b)
        a = bmg.add_addition(s2, t2)
        sample_function_2(s1, a)
        observed = bmg.to_dot()
        expected = """
digraph "graph" {
  N0[label=2.0];
  N10[label=Bernoulli];
  N11[label=Sample];
  N12[label=3.0];
  N13[label=map];
  N14[label=index];
  N15[label=0.0];
  N16[label=Sample];
  N17[label=0.03999999910593033];
  N18[label=Bernoulli];
  N19[label=Sample];
  N1[label=0.5];
  N20[label=map];
  N21[label=index];
  N22[label=1.0];
  N23[label=map];
  N24[label=index];
  N2[label=Bernoulli];
  N3[label=Sample];
  N4[label=Sample];
  N5[label="+"];
  N6[label=0.019999999552965164];
  N7[label=Bernoulli];
  N8[label=Sample];
  N9[label=0.029999999329447746];
  N10 -> N9[label=probability];
  N11 -> N10[label=operand];
  N13 -> N0[label=0];
  N13 -> N11[label=3];
  N13 -> N12[label=2];
  N13 -> N8[label=1];
  N14 -> N13[label=left];
  N14 -> N5[label=right];
  N16 -> N10[label=operand];
  N18 -> N17[label=probability];
  N19 -> N18[label=operand];
  N2 -> N1[label=probability];
  N20 -> N0[label=0];
  N20 -> N12[label=2];
  N20 -> N16[label=1];
  N20 -> N19[label=3];
  N21 -> N20[label=left];
  N21 -> N5[label=right];
  N23 -> N14[label=1];
  N23 -> N15[label=0];
  N23 -> N21[label=3];
  N23 -> N22[label=2];
  N24 -> N23[label=left];
  N24 -> N3[label=right];
  N3 -> N2[label=operand];
  N4 -> N2[label=operand];
  N5 -> N0[label=right];
  N5 -> N4[label=left];
  N7 -> N6[label=probability];
  N8 -> N7[label=operand];
}
        """
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())
