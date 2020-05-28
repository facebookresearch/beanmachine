# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for rules.py"""
import ast
import re
import unittest
from typing import Any

import astor
from beanmachine.ppl.utils.ast_patterns import (
    add,
    ast_domain,
    ast_false,
    ast_true,
    attribute,
    binop,
    constant_tensor_any,
    expr,
    function_def,
    name,
    num,
)
from beanmachine.ppl.utils.patterns import (
    ListAny,
    PredicatePattern,
    anyPattern as _default,
    match_every,
)
from beanmachine.ppl.utils.rules import (
    AllOf,
    ListEdit,
    PatternRule,
    SomeOf,
    TryMany as many,
    TryOnce as once,
    at_least_once,
    either_or_both,
    fail,
    if_then,
    ignore_div_zero,
    ignore_runtime_error,
    list_member_children,
    make_logger,
    pattern_rules,
    projection_rule,
    remove_from_list,
)


def tidy(s: str) -> str:
    return re.sub(" +", " ", s.replace("\n", " ")).strip()


def first_expr(s: str) -> ast.AST:
    return ast.parse(s).body[0].value


_all = ast_domain.all_children
some = ast_domain.some_children
one = ast_domain.one_child
top_down = ast_domain.top_down
bottom_up = ast_domain.bottom_up
descend_until = ast_domain.descend_until
specific_child = ast_domain.specific_child


class RulesTest(unittest.TestCase):
    def test_rules_1(self) -> None:
        """Tests for rules.py"""

        remove_plus_zero = pattern_rules(
            [
                (binop(op=add, left=num(n=0)), lambda b: b.right),
                (binop(op=add, right=num(n=0)), lambda b: b.left),
            ],
            "remove_plus_zero",
        )

        self.maxDiff = None

        m = ast.parse("0; 1; 1+1; 0+1; 1+0; 0+1+0; 0+(1+0); (0+1)+(1+0)")
        # z = m.body[0].value
        o = m.body[1].value
        oo = m.body[2].value
        zo = m.body[3].value
        oz = m.body[4].value
        zo_z = m.body[5].value
        z_oz = m.body[6].value
        zo_oz = m.body[7].value

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
            lambda b: ListEdit([ast.Expr(b.value.left), ast.Expr(b.value.right)]),
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

        # Test top-down and bottom-up combinators:

        # The top-down and bottom-up combinators recursively apply a rule to every
        # node in a tree; top-down rewrites the root and then rewrites all the new
        # children; bottom-up rewrites the leaves and then the new parents.

        # What is the difference between bottom-up and top-down traversals?
        # Consider this example.

        test = ast.parse("m(0, 1, 2+3, [0+4, 5+0, 0+6+0], {0+(7+0): (0+8)+(9+0)})")
        expected = ast.parse("m(0, 1, 2+3, [4, 5, 6], {7: 8+9})")
        result = bottom_up(rpz_once)(test).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(expected))

        # As we'd expect, the bottom-up traversal eliminates all the +0 operations
        # from the tree. But top-down does not!

        result = top_down(rpz_once)(test).expect_success()
        expected = ast.parse("m(0, 1, 2+3, [4, 5, 0+6], {7+0: 8+9})")
        self.assertEqual(ast.dump(result), ast.dump(expected))

        # Why are 0+6+0 and 0+(7+0) not simplified to 6 and 7 by top_down?
        # Well, think about what top-down does when it encounters 0+6+0.
        # First it notes that 0+6+0 has the form x+0 and simplifies it to x,
        # so we have 0+6.  Then we recurse on the children, but the children
        # are not of the form 0+x or x+0, so we're done.  I said rpz_once,
        # not rpz_many, which would keep trying to simplify until proceding
        # to the children:

        result = top_down(rpz_many)(test).expect_success()
        expected = ast.parse("m(0, 1, 2+3, [4, 5, 6], {7: 8+9})")
        self.assertEqual(ast.dump(result), ast.dump(expected))

        # The at_least_once combinator requires that a rule succeed at least
        # once, and then runs it until it fails.

        alorpz = at_least_once(remove_plus_zero)
        self.assertFalse(alorpz(o))
        self.assertTrue(alorpz(z_oz))

    def test_infinite_loop_detection(self) -> None:
        # While working on a previous test case I accidentally created a pattern
        # that has an infinite loop; one of the benefits of a combinator-based
        # approach to rewriting is we can often detect statically when a particular
        # combination of rules must produce an infinite loop, and raise an error.

        # In particular, we know several of the rules always succeed (TryMany,
        # TryOnce, identity) and if any of these rules are ever passed to TryMany,
        # we've got an infinite loop right there.

        # Here's an example. fail always fails, but once always succeeds. Since
        # once always succeeds, _all(once(anything)) always succeeds, which means
        # that we've given something that always succeeds to many, and we'll loop
        # forever.

        with self.assertRaises(ValueError):
            _all = ast_domain.all_children
            _all(many(_all(once(fail))))

    def test_rules_2(self) -> None:
        """Tests for rules.py"""
        self.maxDiff = None
        _all = ast_domain.all_children
        num_stmt = expr(num())
        even = PatternRule(
            match_every(num_stmt, PredicatePattern(lambda e: e.value.n % 2 == 0))
        )
        add_one = projection_rule(lambda e: ast.Expr(ast.Num(e.value.n + 1)))
        t = ast.parse("0; 1; 2; 3; 4; 5 + 6")
        result = _all(_all(if_then(even, add_one)))(t).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(ast.parse("1; 1; 3; 3; 5; 5 + 6")))

    def test_find_random_variables(self) -> None:
        """Find all the functions that have a decorator, delete everything else."""

        self.maxDiff = None
        _all = ast_domain.all_children

        rule = pattern_rules(
            [
                (
                    function_def(
                        decorator_list=ListAny(attribute(attr="random_variable"))
                    ),
                    lambda f: ast.FunctionDef(
                        name=f.name,
                        args=f.args,
                        body=[ast.Pass()],
                        returns=None,
                        decorator_list=[],
                    ),
                ),
                (_default, lambda x: remove_from_list),
            ]
        )
        source = """
# foo.py
@bm.random_variable
def bias() -> Beta:
    return Beta(1, 1)

@bm.random_variable
def toss(i) -> Bernoulli:
    return Bernoulli(bias())

def foo():
    return 123
        """

        expected = """
def bias():
    pass

def toss(i):
    pass
        """
        m = ast.parse(source)
        result = _all(_all(rule))(m).expect_success()
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))

    def test_rules_3(self) -> None:
        """Tests for rules.py"""
        self.maxDiff = None

        # Some nodes, like BoolOp, have the interesting property that they
        # have both regular children and children in a list, which makes it
        # inconvenient to apply a rule to all the "logical" children. This
        # combinator helps with that.

        t = ast.NameConstant(True)
        f = ast.NameConstant(False)

        swap_bools = pattern_rules(
            [(ast_true, lambda n: f), (ast_false, lambda n: t)], "swap_bools"
        )

        # First we'll try it without the combinator:

        _all = ast_domain.all_children

        # "True < False < 1" has this structure:
        c = ast.Compare(ops=[ast.Lt(), ast.Lt()], left=t, comparators=[f, ast.Num(1)])
        result = _all(once(swap_bools))(c).expect_success()

        # all applies the rule to op, left and comparators; since op and comparators
        # do not match the pattern, they're unchanged. But we do not recurse
        # into values, so we only change the first one:

        expected = "(False < False < 1)"
        observed = astor.to_source(result)
        self.assertEqual(observed.strip(), expected.strip())

        # This version treats all the ops and values as children, and as
        # we intend, the rule operates on all the children:

        result = _all(list_member_children(once(swap_bools)))(c).expect_success()
        expected = "(False < True < 1)"
        observed = astor.to_source(result)
        self.assertEqual(observed.strip(), expected.strip())

    def test_rules_4(self) -> None:
        """Tests for rules.py"""
        self.maxDiff = None

        # either_or_both logically takes two rules A and B, and tries to apply
        # Compose(A, B), A, or B, in that order. The first that succeeds is
        # the result.

        zero_to_one = PatternRule(0, lambda n: 1)
        one_to_two = PatternRule(1, lambda n: 2)
        eob = either_or_both(zero_to_one, one_to_two)
        self.assertEqual(eob(0).expect_success(), 2)
        self.assertEqual(eob(1).expect_success(), 2)
        self.assertTrue(eob(2).is_fail())

        # The some_top_down combinator applies a rule to every node in the tree,
        # from root to leaves, but ignores nodes for which the rule fails.
        # It succeeds iff the rule succeeded on any node in the tree. This is
        # useful because it guarantees that if it succeeds, then it did the most
        # work it could do applying a rule to a tree.

        sometd = ast_domain.some_top_down

        result = sometd(eob)(ast.parse("0 + 1 * 2 + 3")).expect_success()
        expected = "2 + 2 * 2 + 3"
        observed = astor.to_source(result)
        self.assertEqual(observed.strip(), expected.strip())

        # If the rule applies to no node, then we fail.
        self.assertTrue(sometd(eob)(result).is_fail())

        # The some_bottom_up combinator is the same as some_top_down but it
        # works from leaves to root instead of root to leaves.

        somebu = ast_domain.some_bottom_up

        result = somebu(eob)(ast.parse("0 + 1 * 2 + 3")).expect_success()
        expected = "2 + 2 * 2 + 3"
        observed = astor.to_source(result)
        self.assertEqual(observed.strip(), expected.strip())

        # If the rule applies to no node, then we fail.
        self.assertTrue(somebu(eob)(result).is_fail())

        # SomeOf extends either_or_both to arbitrarily many rules.
        zero_to_one = PatternRule(0, lambda n: 1)
        one_to_two = PatternRule(1, lambda n: 2)
        three_to_four = PatternRule(3, lambda n: 4)
        so = SomeOf([zero_to_one, one_to_two, three_to_four])
        self.assertEqual(so(0).expect_success(), 2)
        self.assertEqual(so(1).expect_success(), 2)
        self.assertEqual(so(3).expect_success(), 4)
        self.assertTrue(so(2).is_fail())

        # AllOf extends composition to arbitrarily many rRulesTest
        two_to_three = PatternRule(2, lambda n: 3)
        ao1 = AllOf([zero_to_one, one_to_two, two_to_three])
        self.assertEqual(ao1(0).expect_success(), 3)
        self.assertTrue(ao1(1).is_fail())
        ao2 = AllOf([zero_to_one, one_to_two, three_to_four])
        self.assertTrue(ao2(0).is_fail())

    def test_rules_5(self) -> None:
        """Tests for rules.py"""
        self.maxDiff = None

        self.assertTrue(
            constant_tensor_any(first_expr("torch.tensor(1.0)")).is_success()
        )
        self.assertTrue(
            constant_tensor_any(first_expr("tensor([1.0, 2.0])")).is_success()
        )
        self.assertTrue(
            constant_tensor_any(first_expr("tensor([[1,2],[3,4]])")).is_success()
        )
        self.assertTrue(constant_tensor_any(first_expr("torch.tensor(w)")).is_fail())
        self.assertTrue(constant_tensor_any(first_expr("tensor([y, 2.0])")).is_fail())
        self.assertTrue(
            constant_tensor_any(first_expr("tensor([[1,z],[3,4]])")).is_fail()
        )

    def test_rules_6(self) -> None:
        """Tests for rules.py"""

        # Sometimes a rule's projection will fail with an exception through
        # no fault of our own; it can be expensive or impossible to detect
        # a coming exception in some cases. In those cases we can use a combinator
        # which causes rules that throw exceptions to fail rather than throw.

        def always_throws(x: Any):
            raise NotImplementedError()

        self.maxDiff = None

        d = ignore_div_zero(PatternRule([int, int], lambda l: l[0] / l[1]))
        self.assertEquals(d([10, 5]).expect_success(), 2)
        self.assertTrue(d([10, 0]).is_fail())

        n = ignore_runtime_error(PatternRule(int, always_throws))
        self.assertTrue(n(123).is_fail())

    def test_rules_7(self) -> None:
        """Tests for rules.py"""

        # descend_until is a handy combinator that descends through the tree,
        # top down, until a test rule succeeds. It then applies a rule to
        # the nodes that succeeded but does not further recurse down. It does
        # this for all matching nodes in the tree starting from the root.

        self.maxDiff = None

        # replace all 1 with 2, but only in functions decorated with @frob:

        t = PatternRule(function_def(decorator_list=ListAny(name(id="frob"))))
        r = top_down(once(PatternRule(num(1), lambda n: ast.Num(2))))

        s = """
0
1

@frob
def f():
    0
    1

@frob
def g():
    1
    1

def h():
    0
    1
        """

        expected = """
0
1


@frob
def f():
    0
    2


@frob
def g():
    2
    2


def h():
    0
    1
        """

        result = descend_until(t, r)(ast.parse(s)).expect_success()
        observed = astor.to_source(result)
        self.assertEqual(observed.strip(), expected.strip())

    def test_rules_8(self) -> None:
        """Tests for rules.py"""

        # specific_child applies a rule to a specified child of the rule
        # input; the input is required to have such a child. If the rule
        # succeeds then the output is the input with the rewritten child.

        self.maxDiff = None

        # replace all 1 with 2, but only in functions decorated with @frob:

        log = []
        trace = make_logger(log)

        r = trace(
            top_down(
                once(
                    if_then(
                        PatternRule(binop()),
                        trace(
                            specific_child(
                                "left", PatternRule(num(1), lambda n: ast.Num(2))
                            )
                        ),
                    )
                )
            )
        )

        s = "1 + 1 * 1 + 1"
        expected = "2 + 2 * 1 + 1"
        result = r(ast.parse(s)).expect_success()
        observed = astor.to_source(result)
        self.assertEqual(observed.strip(), expected.strip())

        observed = "\n".join(log)
        expected = """
Started top_down
Started specific_child
Finished specific_child
Started specific_child
Finished specific_child
Started specific_child
Finished specific_child
Finished top_down
        """
        self.assertEqual(observed.strip(), expected.strip())

    def test_rules_9(self) -> None:
        """Tests for rules.py"""

        # This demonstrates that a rule that produces a list edit will
        # recursively rewrite that list edit.

        self.maxDiff = None

        # Recursively replace any list of the form [True, [another list]] with
        # the inner list.
        r = top_down(once(PatternRule([True, list], lambda l: ListEdit(l[1]))))
        s = [[True, [1]], [True, [[True, [2]], 3]]]
        expected = "[1, 2, 3]"
        observed = str(r(s).expect_success())
        self.assertEqual(observed.strip(), expected.strip())
