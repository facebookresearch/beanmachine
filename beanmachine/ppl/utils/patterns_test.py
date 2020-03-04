# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for patterns.py"""
import ast
import re
import unittest
from ast import Add, BinOp, Num

from beanmachine.ppl.utils.ast_patterns import ast_str, binop, name_constant, num
from beanmachine.ppl.utils.patterns import negate


def tidy(s: str) -> str:
    return re.sub(" +", " ", s.replace("\n", " ")).strip()


class PatternsTest(unittest.TestCase):
    def test_atomic(self) -> None:
        """Test atomic patterns"""

        p = binop(
            left=binop(left=num(n=0), right=name_constant(value=True)), right=num(n=1.5)
        )
        observed = str(p)

        expected = """
(isinstance(test, BinOp) and
(isinstance(test.left, BinOp) and
(isinstance(test.left.left, Num) and test.left.left.n==0) and
(isinstance(test.left.right, NameConstant) and test.left.right.value==True)) and
(isinstance(test.right, Num) and test.right.n==1.5))
"""
        self.maxDiff = None
        self.assertEqual(tidy(observed), tidy(expected))

        result = p.match(ast.parse("0 * True + 1.5").body[0].value)
        self.assertTrue(result.is_success())
        # This one fails because it is binop(0, binop(true, 1.5)), and the
        # pattern is looking for binop(binop(0, true), 1.5)
        result = p.match(ast.parse("0 + True * 1.5").body[0].value)
        self.assertTrue(result.is_fail())

    def test_negate(self) -> None:
        """Test negate"""

        p = negate(ast_str(s="abc"))
        result = p.match(ast.parse("'abc'").body[0].value)
        self.assertTrue(result.is_fail())
        result = p.match(ast.parse("1+2").body[0].value)
        self.assertTrue(result.is_success())

    def test_1(self) -> None:
        """Tests for patterns.py"""

        b = binop(op=Add, left=num(n=0))
        observed = str(b)

        expected = """
(isinstance(test, BinOp) and isinstance(test.op, Add) and
(isinstance(test.left, Num) and test.left.n==0))
"""
        self.maxDiff = None
        self.assertEqual(tidy(observed), tidy(expected))

        zero = Num(n=0)
        one = Num(n=1)
        result = b.match(BinOp(op=Add(), left=zero, right=one))
        self.assertTrue(result.is_success())
        result = b.match(BinOp(op=Add(), left=one, right=zero))
        self.assertTrue(result.is_fail())
