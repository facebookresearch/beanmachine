# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.compiler.profiler import ProfilerData


class ProfilerTest(unittest.TestCase):
    def test_profiler(self) -> None:
        self.maxDiff = None
        pd = ProfilerData()
        pd.begin("A", 1000000000)
        pd.begin("B", 1100000000)
        pd.begin("C", 1200000000)
        pd.finish("C", 1300000000)
        pd.begin("C", 1400000000)
        pd.finish("C", 1500000000)
        pd.finish("B", 1600000000)
        pd.finish("A", 1700000000)
        pd.begin("D", 1800000000)
        pd.finish("D", 1900000000)

        expected = """
begin A 1000000000
begin B 1100000000
begin C 1200000000
finish C 1300000000
begin C 1400000000
finish C 1500000000
finish B 1600000000
finish A 1700000000
begin D 1800000000
finish D 1900000000"""
        self.assertEqual(expected.strip(), str(pd).strip())

        # B accounts for 500 ms of A;  the two Cs account for 200 ms of B;
        # the rest is unattributed

        report = pd.to_report()

        expected = """
A:(1) 700 ms
  B:(1) 500 ms
    C:(2) 200 ms
    unattributed: 300 ms
  unattributed: 200 ms
D:(1) 100 ms
unattributed: 800 ms
"""
        self.assertEqual(expected.strip(), str(report).strip())

        self.assertEqual(700000000, report.A.total_time)
        self.assertEqual(500000000, report.A.B.total_time)
        self.assertEqual(200000000, report.A.B.C.total_time)
