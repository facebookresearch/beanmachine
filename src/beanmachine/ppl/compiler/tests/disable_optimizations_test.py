# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import scipy
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def sum_1():
    return norm(0) + norm(1) + norm(2)


@bm.functional
def sum_2():
    return norm(3) + norm(4) + norm(5)


@bm.functional
def sum_3():
    return sum_1() + 5.0


@bm.functional
def sum_4():
    return sum_1() + sum_2()


class DisableOptimizationsTest(unittest.TestCase):
    def test_multiary_ops_opt_to_dot(self) -> None:
        self.maxDiff = None
        observations = {}
        queries = [sum_3(), sum_4()]

        skip_optimizations = {"MultiaryOperatorFixer"}
        observed = BMGInference().to_dot(
            queries, observations, skip_optimizations=skip_optimizations
        )

        # Expected model when skipping multiary addition optimization

        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label="+"];
  N07[label="+"];
  N08[label=5.0];
  N09[label="+"];
  N10[label=Query];
  N11[label=Sample];
  N12[label=Sample];
  N13[label=Sample];
  N14[label="+"];
  N15[label="+"];
  N16[label="+"];
  N17[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N02 -> N11;
  N02 -> N12;
  N02 -> N13;
  N03 -> N06;
  N04 -> N06;
  N05 -> N07;
  N06 -> N07;
  N07 -> N09;
  N07 -> N16;
  N08 -> N09;
  N09 -> N10;
  N11 -> N14;
  N12 -> N14;
  N13 -> N15;
  N14 -> N15;
  N15 -> N16;
  N16 -> N17;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

        # Expected graph without skipping multiary addition optimization:

        observed = BMGInference().to_dot(queries, observations)
        expected = """
digraph "graph" {
  N00[label=0.0];
  N01[label=1.0];
  N02[label=Normal];
  N03[label=Sample];
  N04[label=Sample];
  N05[label=Sample];
  N06[label="+"];
  N07[label=5.0];
  N08[label="+"];
  N09[label=Query];
  N10[label=Sample];
  N11[label=Sample];
  N12[label=Sample];
  N13[label="+"];
  N14[label=Query];
  N00 -> N02;
  N01 -> N02;
  N02 -> N03;
  N02 -> N04;
  N02 -> N05;
  N02 -> N10;
  N02 -> N11;
  N02 -> N12;
  N03 -> N06;
  N04 -> N06;
  N05 -> N06;
  N06 -> N08;
  N06 -> N13;
  N07 -> N08;
  N08 -> N09;
  N10 -> N13;
  N11 -> N13;
  N12 -> N13;
  N13 -> N14;
}
"""
        self.assertEqual(expected.strip(), observed.strip())

    def test_multiary_ops_opt_inference(self) -> None:
        observations = {}
        queries = [sum_3(), sum_4()]
        num_samples = 1000

        skip_optimizations = {"MultiaryOperatorFixer"}
        posterior_wo_opt = BMGInference().infer(
            queries, observations, num_samples, skip_optimizations=skip_optimizations
        )
        sum_3_samples_wo_opt = posterior_wo_opt[sum_3()][0]
        sum_4_samples_wo_opt = posterior_wo_opt[sum_4()][0]

        posterior_w_opt = BMGInference().infer(queries, observations, num_samples)

        sum_3_samples_w_opt = posterior_w_opt[sum_3()][0]
        sum_4_samples_w_opt = posterior_w_opt[sum_4()][0]

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(sum_3_samples_wo_opt, sum_3_samples_w_opt).pvalue, 0.05
        )

        self.assertGreaterEqual(
            scipy.stats.ks_2samp(sum_4_samples_wo_opt, sum_4_samples_w_opt).pvalue, 0.05
        )
