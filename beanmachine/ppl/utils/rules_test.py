# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for rules.py"""
import ast
import re
import unittest
from ast import parse

from beanmachine.ppl.utils.ast_patterns import add, ast_domain, binop, num
from beanmachine.ppl.utils.rules import TryMany, TryOnce, pattern_rules


def tidy(s: str) -> str:
    return re.sub(" +", " ", s.replace("\n", " ")).strip()


class RulesTest(unittest.TestCase):
    def test_1(self) -> None:
        """Tests for rules.py"""

        remove_plus_zero = pattern_rules(
            [
                (binop(op=add, left=num(n=0)), lambda b: b.right),
                (binop(op=add, right=num(n=0)), lambda b: b.left),
            ],
            "remove_plus_zero",
        )

        self.maxDiff = None

        m = parse("0; 1; 1+1; 0+1; 1+0; 0+1+0; 0+(1+0); (0+1)+(1+0)")
        # z = m.body[0].value
        o = m.body[1].value
        oo = m.body[2].value
        zo = m.body[3].value
        oz = m.body[4].value
        zo_z = m.body[5].value
        z_oz = m.body[6].value
        zo_oz = m.body[7].value

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

        _all = ast_domain.all_children

        # Note that _all on this list does not recurse down to the
        # children of the list elements. It runs the rule once on
        # each list element, adn that's it.
        result = _all(once)([oo, zo_z, z_oz, zo_oz]).expect_success()
        self.assertEqual(ast.dump(result[0]), ast.dump(oo))
        self.assertEqual(ast.dump(result[1]), ast.dump(zo))
        self.assertEqual(ast.dump(result[2]), ast.dump(oz))
        self.assertEqual(ast.dump(result[3]), ast.dump(zo_oz))

        # Again, this does not recurse to the children. Rather, it keeps
        # running the rule until the pattern fails; that is different than
        # recursing down into the children!
        result = _all(many)([oo, zo_z, z_oz, zo_oz]).expect_success()
        self.assertEqual(ast.dump(result[0]), ast.dump(oo))
        self.assertEqual(ast.dump(result[1]), ast.dump(o))
        self.assertEqual(ast.dump(result[2]), ast.dump(o))
        self.assertEqual(ast.dump(result[3]), ast.dump(zo_oz))

        # Now instead of running the rule on all elements of a list, let's
        # run the rule once on all *children* of a node. Again, this applies the
        # rule just to the children; it does not recurse down into their
        # children, and it does not re-run the rule on the result.
        result = _all(once)(z_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(zo))

        result = _all(once)(zo_z).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(oz))

        result = _all(once)(zo_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(oo))
