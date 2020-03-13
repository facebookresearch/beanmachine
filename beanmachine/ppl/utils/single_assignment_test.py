# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for single_assignment.py"""
import ast
import unittest

import astor
from beanmachine.ppl.utils.fold_constants import fold
from beanmachine.ppl.utils.single_assignment import single_assignment


class SingleAssignmentTest(unittest.TestCase):
    def test_single_assignment(self) -> None:
        """Tests for single_assignment.py"""

        self.maxDiff = None

        source = """
def f():
    if a and b:
        return 1 + ~x + 2
    return 8 * y / (4 * z)
"""
        m = ast.parse(source)
        result = single_assignment(fold(m))
        expected = """
def f():
    if a and b:
        a3 = 3
        a5 = ~x
        r1 = a3 + a5
        return r1
    a6 = 2.0
    a4 = a6 * y
    r2 = a4 / z
    return r2
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())
