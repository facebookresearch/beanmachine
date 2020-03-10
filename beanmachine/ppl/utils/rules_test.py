# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for rules.py"""
import ast
import re
import unittest
from ast import Expr, parse

from beanmachine.ppl.utils.ast_patterns import add, ast_domain, binop, expr, num
from beanmachine.ppl.utils.rules import (
    ListEdit,
    PatternRule,
    TryMany as many,
    TryOnce as once,
    pattern_rules,
    remove_from_list,
)


def tidy(s: str) -> str:
    return re.sub(" +", " ", s.replace("\n", " ")).strip()


class RulesTest(unittest.TestCase):
    def test_rules(self) -> None:
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

        _all = ast_domain.all_children
        some = ast_domain.some_children
        one = ast_domain.one_child

        rpz_once = once(remove_plus_zero)
        rpz_many = many(remove_plus_zero)

        observed = str(rpz_once)
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

        # Note that _all on this list does not recurse down to the
        # children of the list elements. It runs the rule once on
        # each list element, adn that's it.
        result = _all(rpz_once)([oo, zo_z, z_oz, zo_oz]).expect_success()
        self.assertEqual(ast.dump(result[0]), ast.dump(oo))
        self.assertEqual(ast.dump(result[1]), ast.dump(zo))
        self.assertEqual(ast.dump(result[2]), ast.dump(oz))
        self.assertEqual(ast.dump(result[3]), ast.dump(zo_oz))

        # Again, this does not recurse to the children. Rather, it keeps
        # running the rule until the pattern fails; that is different than
        # recursing down into the children!
        result = _all(rpz_many)([oo, zo_z, z_oz, zo_oz]).expect_success()
        self.assertEqual(ast.dump(result[0]), ast.dump(oo))
        self.assertEqual(ast.dump(result[1]), ast.dump(o))
        self.assertEqual(ast.dump(result[2]), ast.dump(o))
        self.assertEqual(ast.dump(result[3]), ast.dump(zo_oz))

        # Now instead of running the rule on all elements of a list, let's
        # run the rule once on all *children* of a node. Again, this applies the
        # rule just to the children; it does not recurse down into their
        # children, and it does not re-run the rule on the result.
        result = _all(rpz_once)(z_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(zo))

        result = _all(rpz_once)(zo_z).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(oz))

        result = _all(rpz_once)(zo_oz).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(oo))

        # Above we had a test for _all(many(rpz))([oo, zo_z, z_oz, zo_oz]);
        # we can get the same results with:
        result = many(some(remove_plus_zero))([oo, zo_z, z_oz, zo_oz]).expect_success()
        self.assertEqual(ast.dump(result[0]), ast.dump(oo))
        self.assertEqual(ast.dump(result[1]), ast.dump(o))
        self.assertEqual(ast.dump(result[2]), ast.dump(o))
        self.assertEqual(ast.dump(result[3]), ast.dump(zo_oz))

        # Both attain a fixpoint.

        # OneChild applies a rule to members of a list or the children of a node,
        # until the first success, and then it stops.
        result = one(remove_plus_zero)([oo, zo_z, z_oz, zo_oz]).expect_success()
        self.assertEqual(ast.dump(result[0]), ast.dump(oo))  # Rule fails
        self.assertEqual(ast.dump(result[1]), ast.dump(zo))  # Rule succeeds
        self.assertEqual(ast.dump(result[2]), ast.dump(z_oz))  # Rule does not run
        self.assertEqual(ast.dump(result[3]), ast.dump(zo_oz))  # Rule does not run

        # Testing list editing:

        # Let's start with a simple test: remove all the zeros from a list
        # of integers:

        remove_zeros = PatternRule(0, lambda b: remove_from_list, "remove_zeros")
        result = _all(once(remove_zeros))(
            [0, 1, 0, 2, 0, 0, 3, 4, 0, 0]
        ).expect_success()
        self.assertEqual(result, [1, 2, 3, 4])

        # Let's try some deeper combinations. Here we apply a rule to all
        # children of a module -- that is, the body. That rule then applies
        # remove_num_statements once to all members of the body list.

        remove_num_statements = PatternRule(
            expr(num()), lambda b: remove_from_list, "remove_num_statements"
        )
        t = ast.parse("0; 1; 2 + 3; 4 + 5 + 6; 7 + 8 * 9;")
        result = _all(_all(once(remove_num_statements)))(t).expect_success()
        self.assertEqual(
            ast.dump(result), ast.dump(ast.parse("2 + 3; 4 + 5 + 6; 7 + 8 * 9;"))
        )

        # Split every statement that is a binop into two statements,
        # and keep going until you can split no more:

        split_binops = PatternRule(
            expr(binop()),
            lambda b: ListEdit([Expr(b.value.left), Expr(b.value.right)]),
            "split_binops",
        )

        # This correctly implements those semantics.
        # The "some" fails when no more work can be done, so the "many"
        # repeats until a fixpoint is reached for the statement list.

        result = _all(many(some(split_binops)))(t).expect_success()
        self.assertEqual(
            ast.dump(result), ast.dump(ast.parse("0; 1; 2; 3; 4; 5; 6; 7; 8; 9;"))
        )

        # TODO: Unfortunately, this does not attain a fixpoint.
        # TODO: This seems like it should have the same behaviour as the
        # TODO: previous, but what happens is:  split_binops returns a ListEdit.
        # TODO: TryMany then checks whether split_binops applies again;
        # TODO: it does not because a ListEdit is not an Expr(BinOp); it is a
        # TODO: ListEdit possibly containing an Expr(BinOp).  It then returns the
        # TODO: ListEdit to AllChildren, which splices in the result and goes on
        # TODO: to the next item in the list.
        # TODO:
        # TODO: We have a problem here: should rules which return ListEdits to other
        # TODO: rules have those other rules automatically distribute their behaviour
        # TODO: across the ListEdit?  Should we disallow a rule from returning a
        # TODO: ListEdit to anything other than All/Some/One, which are the only
        # TODO: combinators that know how to splice in the edit?  Give this some
        # TODO:  thought.
        result = _all(_all(many(split_binops)))(t).expect_success()
        self.assertEqual(
            ast.dump(result), ast.dump(ast.parse("0; 1; 2; 3; 4 + 5; 6; 7; 8 * 9;"))
        )
