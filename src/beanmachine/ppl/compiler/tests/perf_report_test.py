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


def tidy(s):
    s = re.sub(r"generated_at:.*\n", "generated_at: --\n", s)
    s = re.sub(r"\d+ ms", "-- ms", s)
    s = re.sub(r"\(\d+\)", "(--)", s)
    return s


class PerfReportTest(unittest.TestCase):
    def test_bmg_performance_report_1(self) -> None:
        # How to obtain the performance report from BMGInference

        self.maxDiff = None
        queries = [coin()]
        observations = {flip(): tensor(1.0)}
        num_samples = 1000

        # We have an _infer method which returns both samples and a
        # performance report.

        _, report = BMGInference()._infer(queries, observations, num_samples)

        # You can convert the report to a string:

        observed = str(report)
        expected = """
title: Bean Machine Graph performance report
generated_at: --
num_samples: 1000
algorithm: 3
seed: 5123401
node_count: 5
edge_count: 5
factor_count: 0
dist_count: 2
const_count: 1
op_count: 2
add_count: 0
det_supp_count: [0]
bmg_profiler_report: nmc_infer:(1) -- ms
  initialize:(1) -- ms
  collect_samples:(1) -- ms
    step:(1000) -- ms
      save_old:(1000) -- ms
      compute_grads:(2000) -- ms
      create_prop:(2000) -- ms
      sample:(1000) -- ms
      eval:(1000) -- ms
      clear_grads:(1000) -- ms
      restore_old:(7) -- ms
      unattributed: -- ms
    collect_sample:(1000) -- ms
    unattributed: -- ms
  unattributed: -- ms

profiler_report: accumulate:(1) -- ms
infer:(1) -- ms
  import_fix_problems:(--) -- ms
  fix_problems:(1) -- ms
    TensorOpsFixer:(1) -- ms
    AdditionFixer:(1) -- ms
    BoolComparisonFixer:(1) -- ms
    UnsupportedNodeFixer:(1) -- ms
    MultiaryOperatorFixer:(1) -- ms
    RequirementsFixer:(1) -- ms
    ObservationsFixer:(1) -- ms
    unattributed: -- ms
  build_bmg_graph:(1) -- ms
  graph_infer:(1) -- ms
  deserialize_perf_report:(--) -- ms
  transpose_samples:(1) -- ms
  build_mcsamples:(1) -- ms
  unattributed: -- ms
        """

        # Note that there are two profiler reports: one for time spent
        # in the compiler and one for time spent in BMG inference.
        #
        # See next test for details of how to access the elements of the
        # perf report and the profile reports

        self.assertEqual(tidy(expected).strip(), tidy(observed).strip())

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

        # You can dump the entire report as a string. Notice that this
        # version of the report does not include beanstalk compiler timings
        # because of course we did not run the compiler in this test.

        observed = str(perf_report)
        expected = """
title: Bean Machine Graph performance report
generated_at: --
num_samples: 1000
algorithm: 3
seed: 5123401
node_count: 5
edge_count: 5
factor_count: 0
dist_count: 2
const_count: 1
op_count: 2
add_count: 0
det_supp_count: [0]
bmg_profiler_report: nmc_infer:(1) -- ms
  initialize:(1) -- ms
  collect_samples:(1) -- ms
    step:(1000) -- ms
      save_old:(1000) -- ms
      compute_grads:(2000) -- ms
      create_prop:(2000) -- ms
      sample:(1000) -- ms
      eval:(1000) -- ms
      clear_grads:(1000) -- ms
      restore_old:(7) -- ms
      unattributed: -- ms
    collect_sample:(1000) -- ms
    unattributed: -- ms
  unattributed: -- ms
        """
        self.assertEqual(tidy(expected).strip(), tidy(observed).strip())

        # Of you can look at each element programmatically:

        self.assertEqual("Bean Machine Graph performance report", perf_report.title)
        self.assertEqual(3, perf_report.algorithm)
        self.assertEqual(num_samples, perf_report.num_samples)
        self.assertEqual(5, perf_report.node_count)
        self.assertEqual(2, perf_report.dist_count)
        self.assertEqual(1, perf_report.const_count)
        self.assertEqual(0, perf_report.factor_count)
        self.assertEqual(2, perf_report.op_count)
        self.assertEqual(0, perf_report.add_count)
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

        # Or you can dump just the profiler report as a string.
        observed = str(prof_report)

        expected = """
nmc_infer:(1) -- ms
  initialize:(1) -- ms
  collect_samples:(1) -- ms
    step:(1000) -- ms
      save_old:(1000) -- ms
      compute_grads:(2000) -- ms
      create_prop:(2000) -- ms
      sample:(1000) -- ms
      eval:(1000) -- ms
      clear_grads:(1000) -- ms
      restore_old:(7) -- ms
      unattributed: -- ms
    collect_sample:(1000) -- ms
    unattributed: -- ms
  unattributed: -- ms"""

        self.assertEqual(tidy(expected).strip(), tidy(observed).strip())
