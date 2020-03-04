# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for rules.py"""
import ast
import re
import unittest
from ast import Add, BinOp, Num

from beanmachine.ppl.utils.ast_patterns import binop, num
from beanmachine.ppl.utils.rules import TryMany, TryOnce, pattern_rules


def tidy(s: str) -> str:
    return re.sub(" +", " ", s.replace("\n", " ")).strip()


class RulesTest(unittest.TestCase):
    def test_1(self) -> None:
        """Tests for rules.py"""

        remove_plus_zero = pattern_rules(
            [
                (binop(op=Add, left=num(n=0)), lambda b: b.right),
                (binop(op=Add, right=num(n=0)), lambda b: b.left),
            ],
            "remove_plus_zero",
        )

        self.maxDiff = None

        z = Num(n=0)
        o = Num(n=1)
        oo = BinOp(op=Add(), left=o, right=o)
        zo = BinOp(op=Add(), left=z, right=o)
        oz = BinOp(op=Add(), left=o, right=z)
        zo_z = BinOp(op=Add(), left=zo, right=z)
        z_oz = BinOp(op=Add(), left=z, right=oz)
        zo_oz = BinOp(op=Add(), left=zo, right=oz)

        once = TryOnce(remove_plus_zero)
        many = TryMany(remove_plus_zero)

        observed = str(once)
        expected = """
try_once(
  first_match(
    remove_plus_zero(
      (isinstance(test, BinOp) and
      isinstance(test.op, Add) and
      (isinstance(test.left, Num) and test.left.n==0)),
    remove_plus_zero(
      (isinstance(test, BinOp) and
      isinstance(test.op, Add) and
      (isinstance(test.right, Num) and test.right.n==0)) ) )
"""
        self.assertEqual(tidy(observed), tidy(expected))

        result = once(oo).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(oo))

        result = once(zo_z).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(zo))

        result = once(z_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(oz))

        result = many(z_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(o))

        result = many(zo_z).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(o))

        # Does not recurse!
        result = many(zo_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(zo_oz))
