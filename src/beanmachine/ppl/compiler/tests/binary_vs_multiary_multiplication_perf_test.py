# Copyright (c) Facebook, Inc. and its affiliates.
"""Test performance of multiary multiplication optimization """
import random
import re
import unittest

import beanmachine.ppl as bm
import torch
from beanmachine.ppl.inference import BMGInference
from torch.distributions import Normal


@bm.random_variable
def norm(x):
    return Normal(0.0, 1.0)


@bm.functional
def prod_1(counter):
    prod = 1.0
    for i in range(counter):
        prod = prod * norm(i)
    return prod


@bm.functional
def prod_2():
    return prod_1(100) * prod_1(50)


def get_report(skip_optimizations):
    observations = {}
    queries = [prod_2()]
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


class BinaryVsMultiaryMultiplicationPerformanceTest(unittest.TestCase):
    def test_perf_num_nodes_edges(self) -> None:
        """
        Test to check if Multiary multiplication optimization reduces the
        number of nodes and number of edges using the performance
        report returned by BMGInference.
        We initialize the seed to ensure the test is deterministic.
        """
        self.maxDiff = None
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)

        skip_optimizations = set()
        report_w_optimization = get_report(skip_optimizations)

        observed_report_w_optimization = str(report_w_optimization)
        expected_report_w_optimization = """
title: Bean Machine Graph performance report
generated_at: --
num_samples: 1000
algorithm: 3
seed: 5123401
node_count: 105
edge_count: 204
factor_count: 0
dist_count: 1
const_count: 2
op_count: 102
add_count: 0
det_supp_count: [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
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

profiler_report: accumulate:(1) -- ms
infer:(1) -- ms
  fix_problems:(1) -- ms
    BoolArithmeticFixer:(1) -- ms
    AdditionFixer:(1) -- ms
    BoolComparisonFixer:(1) -- ms
    UnsupportedNodeFixer:(1) -- ms
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
"""

        self.assertEqual(
            tidy(observed_report_w_optimization).strip(),
            tidy(expected_report_w_optimization).strip(),
        )

        skip_optimizations = {"MultiaryMultiplicationFixer"}
        report_wo_optimization = get_report(skip_optimizations)

        observed_report_wo_optimization = str(report_wo_optimization)

        expected_report_wo_optimization = """
title: Bean Machine Graph performance report
generated_at: --
num_samples: 1000
algorithm: 3
seed: 5123401
node_count: 203
edge_count: 302
factor_count: 0
dist_count: 1
const_count: 2
op_count: 200
add_count: 0
det_supp_count: [100,100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
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

profiler_report: accumulate:(1) -- ms
infer:(1) -- ms
  fix_problems:(1) -- ms
    BoolArithmeticFixer:(1) -- ms
    AdditionFixer:(1) -- ms
    BoolComparisonFixer:(1) -- ms
    UnsupportedNodeFixer:(1) -- ms
    MultiaryAdditionFixer:(1) -- ms
    LogSumExpFixer:(1) -- ms
    RequirementsFixer:(1) -- ms
    ObservationsFixer:(1) -- ms
    unattributed: -- ms
  build_bmg_graph:(1) -- ms
  graph_infer:(1) -- ms
  deserialize_perf_report:(1) -- ms
  transpose_samples:(1) -- ms
  build_mcsamples:(1) -- ms
  unattributed: -- ms
"""

        self.assertEqual(
            tidy(observed_report_wo_optimization).strip(),
            tidy(expected_report_wo_optimization).strip(),
        )

        expected_nodes_reduction = 98
        observed_nodes_reduction = (
            report_wo_optimization.node_count - report_w_optimization.node_count
        )
        self.assertEqual(expected_nodes_reduction, observed_nodes_reduction)

        expected_edges_reduction = 98
        observed_edges_reduction = (
            report_wo_optimization.edge_count - report_w_optimization.edge_count
        )
        self.assertEqual(expected_edges_reduction, observed_edges_reduction)
