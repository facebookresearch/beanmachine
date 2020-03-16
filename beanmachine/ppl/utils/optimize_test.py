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
