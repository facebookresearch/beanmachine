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
        _, report = BMGInference()._infer(queries, observations, 1000)
        expected = """Bean Machine Graph performance report"""
        self.assertEqual(expected.strip(), report.strip())
