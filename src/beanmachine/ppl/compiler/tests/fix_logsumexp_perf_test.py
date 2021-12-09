# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import platform
import random
import re
import unittest

import beanmachine.ppl as bm
import torch
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


def tidy(s):
    s = re.sub(r"generated_at:.*\n", "generated_at: --\n", s)
    s = re.sub(r"\d+ ms", "-- ms", s)
    s = re.sub(r"\(\d+\)", "(--)", s)
    return s


class LogSumExpPerformanceTest(unittest.TestCase):
    def test_perf_num_nodes_edges(self) -> None:
        """
        Test to check if LogSumExp Transformation reduces the
        number of nodes and number of edges using the performance
        report returned by BMGInference.
        We initialize the seed to ensure the test is deterministic.
        """
        if platform.system() == "Windows":
            self.skipTest("Disabling *_perf_test.py until flakiness is resolved")

        self.maxDiff = None
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        skip_optimizations = {
            "BetaBernoulliConjugateFixer",
            "BetaBinomialConjugateFixer",
            "NormalNormalConjugateFixer",
        }
        report_w_optimization = get_report(skip_optimizations)

        observed_report_w_optimization = str(report_w_optimization)
        expected_report_w_optimization = """
title: Bean Machine Graph performance report
generated_at: --
num_samples: 1000
algorithm: 3
seed: 5123401
node_count: 104
edge_count: 202
factor_count: 0
dist_count: 1
const_count: 2
op_count: 101
add_count: 0
det_supp_count: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
bmg_profiler_report: nmc_infer:(1) -- ms
  initialize:(1) -- ms
  collect_samples:(1) -- ms
    step:(100000) -- ms
      create_prop:(200000) -- ms
        compute_grads:(200000) -- ms
        unattributed: -- ms
      sample:(100000) -- ms
      save_old:(100000) -- ms
      eval:(100000) -- ms
      clear_grads:(100000) -- ms
      restore_old:(8136) -- ms
      unattributed: -- ms
    collect_sample:(1000) -- ms
    unattributed: -- ms
  unattributed: -- ms
unattributed: -- ms

profiler_report: accumulate:(1) -- ms
infer:(1) -- ms
  fix_problems:(1) -- ms
    VectorizedModelFixer:(--) -- ms
    BoolArithmeticFixer:(1) -- ms
    AdditionFixer:(1) -- ms
    BoolComparisonFixer:(1) -- ms
    UnsupportedNodeFixer:(1) -- ms
    MatrixScaleFixer:(--) -- ms
    MultiaryAdditionFixer:(1) -- ms
    LogSumExpFixer:(1) -- ms
    MultiaryMultiplicationFixer:(1) -- ms
    RequirementsFixer:(1) -- ms
    ObservationsFixer:(1) -- ms
    unattributed: -- ms
  build_bmg_graph:(1) -- ms
  graph_infer:(1) -- ms
  deserialize_perf_report:(1) -- ms
  transpose_samples:(1) -- ms
  build_mcsamples:(1) -- ms
  unattributed: -- ms
unattributed: -- ms
"""

        self.assertEqual(
            tidy(observed_report_w_optimization).strip(),
            tidy(expected_report_w_optimization).strip(),
        )

        skip_optimizations = {
            "LogSumExpFixer",
            "BetaBernoulliConjugateFixer",
            "BetaBinomialConjugateFixer",
            "NormalNormalConjugateFixer",
        }
        report_wo_optimization = get_report(skip_optimizations)

        observed_report_wo_optimization = str(report_wo_optimization)
        expected_report_wo_optimization = """
title: Bean Machine Graph performance report
generated_at: --
num_samples: 1000
algorithm: 3
seed: 5123401
node_count: 205
edge_count: 303
factor_count: 0
dist_count: 1
const_count: 2
op_count: 202
add_count: 1
det_supp_count: [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
bmg_profiler_report: nmc_infer:(1) -- ms
  initialize:(1) -- ms
  collect_samples:(1) -- ms
    step:(100000) -- ms
      create_prop:(200000) -- ms
        compute_grads:(200000) -- ms
        unattributed: -- ms
      sample:(100000) -- ms
      save_old:(100000) -- ms
      eval:(100000) -- ms
      clear_grads:(100000) -- ms
      restore_old:(8136) -- ms
      unattributed: -- ms
    collect_sample:(1000) -- ms
    unattributed: -- ms
  unattributed: -- ms
unattributed: -- ms

profiler_report: accumulate:(1) -- ms
infer:(1) -- ms
  fix_problems:(1) -- ms
    VectorizedModelFixer:(--) -- ms
    BoolArithmeticFixer:(1) -- ms
    AdditionFixer:(1) -- ms
    BoolComparisonFixer:(1) -- ms
    UnsupportedNodeFixer:(1) -- ms
    MatrixScaleFixer:(--) -- ms
    MultiaryAdditionFixer:(1) -- ms
    MultiaryMultiplicationFixer:(1) -- ms
    RequirementsFixer:(1) -- ms
    ObservationsFixer:(1) -- ms
    unattributed: -- ms
  build_bmg_graph:(1) -- ms
  graph_infer:(1) -- ms
  deserialize_perf_report:(1) -- ms
  transpose_samples:(1) -- ms
  build_mcsamples:(1) -- ms
  unattributed: -- ms
unattributed: -- ms
"""

        self.assertEqual(
            tidy(observed_report_wo_optimization).strip(),
            tidy(expected_report_wo_optimization).strip(),
        )

        expected_nodes_reduction = 101
        observed_nodes_reduction = (
            report_wo_optimization.node_count - report_w_optimization.node_count
        )
        self.assertEqual(expected_nodes_reduction, observed_nodes_reduction)

        expected_edges_reduction = 101
        observed_edges_reduction = (
            report_wo_optimization.edge_count - report_w_optimization.edge_count
        )
        self.assertEqual(expected_edges_reduction, observed_edges_reduction)
