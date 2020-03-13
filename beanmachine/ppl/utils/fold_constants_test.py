# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for fold_constants.py"""
import ast
import unittest

from beanmachine.ppl.utils.fold_constants import fold


class ConstantFoldTest(unittest.TestCase):
    def test_constant_fold(self) -> None:
        """Tests for fold_constants.py"""

        self.maxDiff = None

        source = """
~0; -1; +2; 3+4*5-6**2; 5&6; 7|8; 9^10
11/0; 12/4; 14%0; 15%5; 4>>1; 4<<1"""
        m = ast.parse(source)
        result = fold(m)
        expected = """
-1; -1; 2; -13; 4; 15; 3
11/0; 3.0; 14%0; 0; 2; 8"""
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))

        source = """
False or False or False or False
False or True or False or False
True and True and True and True
True and False and False and True
x = 1 if not ((True or False) and True) else 2
"""
        m = ast.parse(source)
        result = fold(m)
        expected = "False; True; True; False; x = 2"
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))
