# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for error_report.py"""
import unittest

from beanmachine.ppl.compiler.error_report import ErrorReport, Violation
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder


class ErrorReportTest(unittest.TestCase):
    def test_error_report(self) -> None:
        """test_error_report"""
        bmg = BMGraphBuilder()
        r = bmg.add_real(-2.5)
        b = bmg.add_bernoulli(r)
        v = Violation(r, b.requirements[0], b, b.edges[0])
        e = ErrorReport()
        e.add_error(v)
        expected = """
The probability of a Bernoulli is required to be a probability but is a negative real."""
        self.assertEqual(expected.strip(), str(e).strip())
