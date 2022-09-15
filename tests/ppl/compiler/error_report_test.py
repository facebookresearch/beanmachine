# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for error_report.py"""
import unittest

from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.bmg_types import NegativeReal, Probability
from beanmachine.ppl.compiler.error_report import ErrorReport, Violation


class ErrorReportTest(unittest.TestCase):
    def test_error_report(self) -> None:
        """test_error_report"""
        bmg = BMGraphBuilder()
        r = bmg.add_real(-2.5)
        b = bmg.add_bernoulli(r)
        v = Violation(r, NegativeReal, Probability, b, "probability", {})
        e = ErrorReport()
        e.add_error(v)
        expected = """
The probability of a Bernoulli is required to be a probability but is a negative real."""
        self.assertEqual(expected.strip(), str(e).strip())
