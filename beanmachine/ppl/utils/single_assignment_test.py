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
    for x in [[10, 20], [30, 40]]:
        for y in x:
            print(x + y)
    return 8 * y / (4 * z)
"""
        m = ast.parse(source)
        result = single_assignment(fold(m))
        expected = """
def f():
    if a and b:
        a7 = 3
        a12 = ~x
        a4 = a7 + a12
        a13 = 5
        a16 = 6
        a8 = g(a13, a16)
        r1 = a4 + a8
        return r1
    z = tensor([3.0, 4.0])
    a9 = 10
    a14 = 20
    a5 = [a9, a14]
    a15 = 30
    a17 = 40
    a10 = [a15, a17]
    f2 = [a5, a10]
    for x in f2:
        for y in x:
            print(x + y)
    a11 = 2.0
    a6 = a11 * y
    r3 = a6 / z
    return r3
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())
