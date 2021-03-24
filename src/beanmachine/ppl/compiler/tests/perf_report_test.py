# Copyright (c) Facebook, Inc. and its affiliates.
import re
import unittest

import beanmachine.graph as graph
import beanmachine.ppl as bm
import beanmachine.ppl.compiler.performance_report as pr
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

    def test_bmg_performance_report_2(self) -> None:
        # How to use the performance reporter calling BMG directly
        # rather than through BMGInference / BMGraphBuilder.

        self.maxDiff = None

        g = graph.Graph()

        # Turn on data collection
        g.collect_performance_data(True)

        # Build a simple model:
        #
        # BETA(2, 2) --> SAMPLE --> BERNOULLI  --> SAMPLE --> observe False
        #

        n0 = g.add_constant_pos_real(2.0)
        n1 = g.add_distribution(
            graph.DistributionType.BETA, graph.AtomicType.PROBABILITY, [n0, n0]
        )
        n2 = g.add_operator(graph.OperatorType.SAMPLE, [n1])
        n3 = g.add_distribution(
            graph.DistributionType.BERNOULLI, graph.AtomicType.BOOLEAN, [n2]
        )
        n4 = g.add_operator(graph.OperatorType.SAMPLE, [n3])
        g.observe(n4, False)
        g.query(n2)

        # Run inference
        num_samples = 1000
        g.infer(num_samples, graph.InferenceType.NMC)

        # Fetch raw perf data (JSON string)

        js = g.performance_report()

        # decode

        perf_report = pr.json_to_perf_report(js)

        # There is not yet a str() function on the report, but you can
        # look at the elements programmatically:

        self.assertEqual("Bean Machine Graph performance report", perf_report.title)
        self.assertEqual(3, perf_report.algorithm)
        self.assertEqual(num_samples, perf_report.num_samples)
        self.assertEqual(5, perf_report.node_count)
        self.assertEqual(5, perf_report.edge_count)

        # You can also look at profiler elements programmatically.
        #
        # Ex: how much time do we spend initializing the inference algorithm
        # data structures?

        prof_report = perf_report.bmg_profiler_report

        self.assertLess(0, prof_report.nmc_infer.total_time)
        self.assertLess(0, prof_report.nmc_infer.initialize.total_time)

        # How many times did we do a step?

        self.assertEqual(1000, prof_report.nmc_infer.collect_samples.step.calls)

        # Or you can dump the profiler report as a string.
        s = str(prof_report)
        observed = re.sub(r"\d+ ms", "-- ms", s)

        expected = """
nmc_infer:(1) -- ms
  initialize:(1) -- ms
  collect_samples:(1) -- ms
    step:(1000) -- ms
    unattributed: -- ms
  unattributed: -- ms
        """

        self.assertEqual(expected.strip(), observed.strip())
