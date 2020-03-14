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
        return 1 + ~x + 2 + g(5, 6)
    z = tensor([1.0 + 2.0, 4.0])
    return 8 * y / (4 * z)
"""
        m = ast.parse(source)
        result = single_assignment(fold(m))
        expected = """
def f():
    if a and b:
        a5 = 3
        a8 = ~x
        a3 = a5 + a8
        a9 = 5
        a10 = 6
        a6 = g(a9, a10)
        r1 = a3 + a6
        return r1
    z = tensor([3.0, 4.0])
    a7 = 2.0
    a4 = a7 * y
    r2 = a4 / z
    return r2
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())
