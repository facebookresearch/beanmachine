# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import platform
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import exp, log
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def sum_1(counter):
    sum = 0.0
    for i in range(counter):
        sum = sum + exp(norm(i))
    return sum


@bm.functional
def sum_2():
    return log(sum_1(100))


def get_report(skip_optimizations):
    observations = {}
    queries = [sum_2()]
    number_samples = 1000

    _, perf_report = BMGInference()._infer(
        queries, observations, number_samples, skip_optimizations=skip_optimizations
    )

    return perf_report


class LogSumExpPerformanceTest(unittest.TestCase):
    def test_perf_num_nodes_edges(self) -> None:
        """
        Test to check if LogSumExp Transformation reduces the
        number of nodes and number of edges using the performance
        report returned by BMGInference.
        """
        if platform.system() == "Windows":
            self.skipTest("Disabling *_perf_test.py until flakiness is resolved")

        self.maxDiff = None

        skip_optimizations = {
            "BetaBernoulliConjugateFixer",
            "BetaBinomialConjugateFixer",
            "NormalNormalConjugateFixer",
        }
        report_w_optimization = get_report(skip_optimizations)
        self.assertEqual(report_w_optimization.node_count, 104)
        self.assertEqual(report_w_optimization.edge_count, 202)

        skip_optimizations = {
            "LogSumExpFixer",
            "BetaBernoulliConjugateFixer",
            "BetaBinomialConjugateFixer",
            "NormalNormalConjugateFixer",
        }
        report_wo_optimization = get_report(skip_optimizations)
        self.assertEqual(report_wo_optimization.node_count, 205)
        self.assertEqual(report_wo_optimization.edge_count, 303)
