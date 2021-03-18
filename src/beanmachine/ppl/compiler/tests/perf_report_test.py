# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
from beanmachine.ppl.inference import BMGInference
from torch import tensor
from torch.distributions import Bernoulli, Beta


@bm.random_variable
def coin():
    return Beta(2.0, 2.0)


@bm.random_variable
def flip():
    return Bernoulli(coin())


class PerfReportTest(unittest.TestCase):
    def test_bmg_performance_report_1(self) -> None:
        self.maxDiff = None
        queries = [coin()]
        observations = {flip(): tensor(1.0)}
        num_samples = 1000
        _, report = BMGInference()._infer(queries, observations, num_samples)

        self.assertEqual("Bean Machine Graph performance report", report.title)
        self.assertEqual(3, report.algorithm)
        self.assertEqual(num_samples, report.num_samples)
        self.assertEqual(5, report.node_count)
        self.assertEqual(5, report.edge_count)
        self.assertLess(0, report.profiler_report.accumulate.total_time)
        self.assertLess(0, report.profiler_report.infer.total_time)
        self.assertLess(0, report.profiler_report.infer.graph_infer.total_time)
        self.assertLess(0, len(report.profiler_data))
        self.assertLess(0, report.profiler_data[0].timestamp)
        self.assertNotEqual("", str(report.bmg_profiler_report))
        self.assertLess(0, report.bmg_profiler_report.nmc_infer.total_time)
        self.assertLess(0, report.bmg_profiler_report.nmc_infer.initialize.total_time)
        self.assertLess(
            0, report.bmg_profiler_report.nmc_infer.collect_samples.total_time
        )
