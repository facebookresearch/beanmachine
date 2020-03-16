# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for optimize.py"""
import ast
import unittest

import astor
from beanmachine.ppl.utils.optimize import optimize


class OptimizeTest(unittest.TestCase):
    def test_optimize_and(self) -> None:
        """Tests for optimize.py"""

        self.maxDiff = None

        source = """
True and x
True and x and y
True and True and x
1 and x
1 and x and y
1 and 1 and x
1.0 and x
1.0 and x and y
1.0 and 1.0 and x
-1 and x
-1 and x and y
-1 and -1 and x
-1.0 and x
-1.0 and x and y
-1.0 and -1.0 and x
tensor(True) and x
tensor(True) and x and y
tensor(True) and tensor(True) and x
tensor(1) and x
tensor(1) and x and y
tensor(1) and tensor(1) and x
tensor(1.0) and x
tensor(1.0) and x and y
tensor(1.0) and tensor(1.0) and x
tensor(-1) and x
tensor(-1) and x and y
tensor(-1) and tensor(-1) and x
tensor(-1.0) and x
tensor(-1.0) and x and y
tensor(-1.0) and tensor(-1.0) and x
tensor([True]) and x
tensor([True]) and x and y
tensor([True]) and tensor([True]) and x
tensor([1]) and x
tensor([1]) and x and y
tensor([1]) and tensor([1]) and x
tensor([1.0]) and x
tensor([1.0]) and x and y
tensor([1.0]) and tensor([1.0]) and x
tensor([-1]) and x
tensor([-1]) and x and y
tensor([-1]) and tensor([-1]) and x
tensor([-1.0]) and x
tensor([-1.0]) and x and y
tensor([-1.0]) and tensor([-1.0]) and x

False and x
0 and x
0.0 and x
-0.0 and x
tensor(False) and x
tensor(0) and x
tensor(0.0) and x
tensor(-0.0) and x
tensor([False]) and x
tensor([0]) and x
tensor([0.0]) and x
tensor([-0.0]) and x
"""
        m = ast.parse(source)
        result = optimize(m)
        expected = """
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
x
x and y
x
False
0
0.0
-0.0
tensor(False)
tensor(0)
tensor(0.0)
tensor(-0.0)
tensor([False])
tensor([0])
tensor([0.0])
tensor([-0.0])"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_optimize_or(self) -> None:
        """Tests for optimize.py"""

        self.maxDiff = None

        source = """
False or x
False or x or y
False or False or x
0 or x
0 or x or y
0 or 0 or x
0.0 or x
0.0 or x or y
0.0 or 0.0 or x
-0 or x
-0 or x or y
-0 or -0 or x
-0.0 or x
-0.0 or x or y
-0.0 or -0.0 or x
tensor(False) or x
tensor(False) or x or y
tensor(False) or tensor(False) or x
tensor(0) or x
tensor(0) or x or y
tensor(0) or tensor(0) or x
tensor(0.0) or x
tensor(0.0) or x or y
tensor(0.0) or tensor(0.0) or x
tensor(-0) or x
tensor(-0) or x or y
tensor(-0) or tensor(-0) or x
tensor(-0.0) or x
tensor(-0.0) or x or y
tensor(-0.0) or tensor(-0.0) or x
tensor([False]) or x
tensor([False]) or x or y
tensor([False]) or tensor([False]) or x
tensor([0]) or x
tensor([0]) or x or y
tensor([0]) or tensor([0]) or x
tensor([0.0]) or x
tensor([0.0]) or x or y
tensor([0.0]) or tensor([0.0]) or x
tensor([-0]) or x
tensor([-0]) or x or y
tensor([-0]) or tensor([-0]) or x
tensor([-0.0]) or x
tensor([-0.0]) or x or y
tensor([-0.0]) or tensor([-0.0]) or x
True or x
1 or x
1.0 or x
-1.0 or x
tensor(True) or x
tensor(1) or x
tensor(1.0) or x
tensor(-1.0) or x
tensor([True]) or x
tensor([1]) or x
tensor([1.0]) or x
tensor([-1.0]) or x
"""
        m = ast.parse(source)
        result = optimize(m)
        expected = """
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
x
x or y
x
True
1
1.0
-1.0
tensor(True)
tensor(1)
tensor(1.0)
tensor(-1.0)
tensor([True])
tensor([1])
tensor([1.0])
tensor([-1.0])"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_optimize_if(self) -> None:
        """Tests for optimize.py"""

        self.maxDiff = None

        source = """
if True or a:
    if False or b:
        1
else:
    2

if False and c:
    if True and d:
        3
else:
    if True or e:
        4
    5
"""
        m = ast.parse(source)
        result = optimize(m)
        expected = """
if b:
    1
4
5
"""
        self.assertEqual(astor.to_source(result).strip(), expected.strip())
