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
  N00[label=0.5];
  N01[label=Bernoulli];
  N02[label=Sample];
  N03[label=0.25];
  N04[label=Bernoulli];
  N05[label=Sample];
  N06[label=0.0];
  N07[label=0.75];
  N08[label=Bernoulli];
  N09[label=Sample];
  N10[label=1.0];
  N11[label=map];
  N12[label=index];
  N01 -> N00[label=probability];
  N02 -> N01[label=operand];
  N04 -> N03[label=probability];
  N05 -> N04[label=operand];
  N08 -> N07[label=probability];
  N09 -> N08[label=operand];
  N11 -> N05[label=1];
  N11 -> N06[label=0];
  N11 -> N09[label=3];
  N11 -> N10[label=2];
  N12 -> N02[label=right];
  N12 -> N11[label=left];
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
  N00[label=2.0];
  N01[label=0.5];
  N02[label=Bernoulli];
  N03[label=Sample];
  N04[label=Sample];
  N05[label="+"];
  N06[label=0.019999999552965164];
  N07[label=Bernoulli];
  N08[label=Sample];
  N09[label=0.029999999329447746];
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
  N20[label=map];
  N21[label=index];
  N22[label=1.0];
  N23[label=map];
  N24[label=index];
  N02 -> N01[label=probability];
  N03 -> N02[label=operand];
  N04 -> N02[label=operand];
  N05 -> N00[label=right];
  N05 -> N04[label=left];
  N07 -> N06[label=probability];
  N08 -> N07[label=operand];
  N10 -> N09[label=probability];
  N11 -> N10[label=operand];
  N13 -> N00[label=0];
  N13 -> N08[label=1];
  N13 -> N11[label=3];
  N13 -> N12[label=2];
  N14 -> N05[label=right];
  N14 -> N13[label=left];
  N16 -> N10[label=operand];
  N18 -> N17[label=probability];
  N19 -> N18[label=operand];
  N20 -> N00[label=0];
  N20 -> N12[label=2];
  N20 -> N16[label=1];
  N20 -> N19[label=3];
  N21 -> N05[label=right];
  N21 -> N20[label=left];
  N23 -> N14[label=1];
  N23 -> N15[label=0];
  N23 -> N21[label=3];
  N23 -> N22[label=2];
  N24 -> N03[label=right];
  N24 -> N23[label=left];
}"""
        self.maxDiff = None
        self.assertEqual(observed.strip(), expected.strip())
