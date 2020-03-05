# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for patterns.py"""
import ast
import re
import unittest

from beanmachine.ppl.utils.ast_patterns import (
    ast_str,
    binop,
    compare,
    expr,
    module,
    name_constant,
    num,
)
from beanmachine.ppl.utils.patterns import ListAny, negate


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

        result = p(ast.parse("0 * True + 1.5").body[0].value)
        self.assertTrue(result.is_success())
        # This one fails because it is binop(0, binop(true, 1.5)), and the
        # pattern is looking for binop(binop(0, true), 1.5)
        result = p(ast.parse("0 + True * 1.5").body[0].value)
        self.assertTrue(result.is_fail())

    def test_negate(self) -> None:
        """Test negate"""

        p = negate(ast_str(s="abc"))
        result = p(ast.parse("'abc'").body[0].value)
        self.assertTrue(result.is_fail())
        result = p(ast.parse("1+2").body[0].value)
        self.assertTrue(result.is_success())

    def test_list_patterns(self) -> None:
        """Tests for list patterns"""

        p = module(body=[])
        observed = str(p)
        expected = """(isinstance(test, Module) and test.body==[])"""
        self.maxDiff = None
        self.assertEqual(tidy(observed), tidy(expected))

        result = p(ast.parse(""))
        self.assertTrue(result.is_success())
        result = p(ast.parse("1 + 2"))
        self.assertTrue(result.is_fail())

        p = module(body=[expr(value=binop()), expr(value=binop())])
        observed = str(p)
        expected = """
(isinstance(test, Module) and
[(isinstance(test.body[0], Expr) and isinstance(test.body[0].value, BinOp)),
(isinstance(test.body[1], Expr) and isinstance(test.body[1].value, BinOp))])
"""
        self.assertEqual(tidy(observed), tidy(expected))

        result = p(ast.parse("1 + 2"))
        self.assertTrue(result.is_fail())
        result = p(ast.parse("1 + 2; 3 * 4"))
        self.assertTrue(result.is_success())

        p = module(ListAny(expr(compare())))
        observed = str(p)
        expected = """
(isinstance(test, Module) and
test.body.any(x:(isinstance(x, Expr) and isinstance(x.value, Compare))))
"""
        self.assertEqual(tidy(observed), tidy(expected))

        result = p(ast.parse("1 + 2; x is None"))
        self.assertTrue(result.is_success())
        result = p(ast.parse("1 + 2; 3 * 4"))
        self.assertTrue(result.is_fail())
