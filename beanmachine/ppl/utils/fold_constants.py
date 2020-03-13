#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast

from beanmachine.ppl.utils.ast_patterns import (
    any_constant,
    ast_domain,
    ast_false,
    ast_true,
    binop,
    bool_constant,
    boolop,
    compare,
    if_exp,
    match_any,
    negative_num,
    non_zero_num,
    num,
    unaryop,
)
from beanmachine.ppl.utils.patterns import (
    HeadTail,
    ListAll,
    Pattern,
    PredicatePattern,
    anyPattern as _any,
    match_every,
    nonEmptyList,
)
from beanmachine.ppl.utils.rules import (
    Compose,
    FirstMatch as first,
    PatternRule,
    Recursive,
    Rule,
    TryMany as many,
    TryOnce as once,
    at_least_once,
    list_member_children,
    pattern_rules,
)


_all = ast_domain.all_children
_bottom_up = ast_domain.bottom_up
_some_top_down = ast_domain.some_top_down

# TODO: Fold operations on constant tensors.
# TODO: Fold matmul?

##########
#
# These operations fold expressions that are entirely constant. Folding expressions
# such as "x and true" to "x" will be a different pass.
#
##########

_fold_arithmetic: Rule = pattern_rules(
    [
        (unaryop(op=ast.USub, operand=num()), lambda u: ast.Num(-u.operand.n)),
        (unaryop(op=ast.UAdd, operand=num()), lambda u: ast.Num(+u.operand.n)),
        (unaryop(op=ast.Invert, operand=num()), lambda u: ast.Num(~u.operand.n)),
        (
            binop(op=ast.Add, left=num(), right=num()),
            lambda b: ast.Num(b.left.n + b.right.n),
        ),
        (
            binop(op=ast.BitAnd, left=num(), right=num()),
            lambda b: ast.Num(b.left.n & b.right.n),
        ),
        (
            binop(op=ast.BitOr, left=num(), right=num()),
            lambda b: ast.Num(b.left.n | b.right.n),
        ),
        (
            binop(op=ast.BitXor, left=num(), right=num()),
            lambda b: ast.Num(b.left.n ^ b.right.n),
        ),
        (
            binop(op=ast.Div, left=num(), right=non_zero_num),
            lambda b: ast.Num(b.left.n / b.right.n),
        ),
        (
            binop(op=ast.LShift, left=num(), right=num()),
            lambda b: ast.Num(b.left.n << b.right.n),
        ),
        (
            binop(op=ast.Mod, left=num(), right=non_zero_num),
            lambda b: ast.Num(b.left.n % b.right.n),
        ),
        (
            binop(op=ast.Mult, left=num(), right=num()),
            lambda b: ast.Num(b.left.n * b.right.n),
        ),
        (
            binop(op=ast.Pow, left=num(), right=num()),
            lambda b: ast.Num(b.left.n ** b.right.n),
        ),
        (
            binop(op=ast.RShift, left=num(), right=num()),
            lambda b: ast.Num(b.left.n >> b.right.n),
        ),
        (
            binop(op=ast.Sub, left=num(), right=num()),
            lambda b: ast.Num(b.left.n - b.right.n),
        ),
    ],
    "fold_arithmetic",
)

_fold_logic: Rule = pattern_rules(
    [
        (
            boolop(op=ast.And, values=ListAll(bool_constant)),
            lambda b: ast.NameConstant(all(v.value for v in b.values)),
        ),
        (
            boolop(op=ast.Or, values=ListAll(bool_constant)),
            lambda b: ast.NameConstant(any(v.value for v in b.values)),
        ),
        (
            unaryop(op=ast.Not, operand=bool_constant),
            lambda u: ast.NameConstant(not u.operand.value),
        ),
    ],
    "fold_logic",
)

_fold_conditional: Rule = pattern_rules(
    [
        (
            if_exp(test=ast_true, body=any_constant, orelse=any_constant),
            lambda i: i.body,
        ),
        (
            if_exp(test=ast_false, body=any_constant, orelse=any_constant),
            lambda i: i.orelse,
        ),
    ],
    "fold_conditional",
)

# If we have "1 < 2" then turn it into True.
# If we have "1 < 2 < 3" then turn it into "1 < 2 and 2 < 3"
_fold_comparison_once: Rule = pattern_rules(
    [
        (
            compare(left=num(), ops=[ast.Eq], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n == c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.Gt], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n > c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.GtE], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n >= c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.Is], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n is c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.IsNot], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n is not c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.Lt], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n < c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.LtE], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n <= c.comparators[0].n),
        ),
        (
            compare(left=num(), ops=[ast.NotEq], comparators=[num()]),
            lambda c: ast.NameConstant(c.left.n != c.comparators[0].n),
        ),
        (
            compare(
                left=num(), ops=HeadTail(_any, nonEmptyList), comparators=ListAll(num())
            ),
            lambda c: ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=c.left, ops=[c.ops[0]], comparators=[c.comparators[0]]
                    ),
                    ast.Compare(
                        left=c.comparators[0],
                        ops=c.ops[1:],
                        comparators=c.comparators[1:],
                    ),
                ],
            ),
        ),
    ],
    "fold_comparison",
)

# A couple notes on this rule construction, since this is our first rule that
# does a recursive list transformation.

# fold_comparison_once produces something that can have the optimization run again
# on its children.  Since we produce a bool_op, which has a list as a child, say
# that we're going to recurse on list members as children. Thus we'll need an
# all(list_member_children) in here somewhere.  But how do we efficiently recurse
# on the new children?

# We don't want to say top_down(_fold_comparison_once) because that then fails
# unless the rule applies to every node in the tree.  We don't want to say
# top_down(once(_fold_comparison_once)) because then we do a complete tree
# traversal every time this rule is applied, which is inefficient.  What we want
# to say is: *if* the rule applies then apply it, and then *attempt* to apply
# the rule again to the *immediate* children.  We'll make a new combinator
# that does this "shallow" top-down traversal for us:


def _shallow_top_down(rule: Rule) -> Rule:
    return Compose(
        rule,
        _all(list_member_children(once(Recursive(lambda: _shallow_top_down(rule))))),
    )


_fold_comparisons: Rule = _shallow_top_down(_fold_comparison_once)


_fold_constants = first(
    [_fold_arithmetic, _fold_logic, _fold_conditional, _fold_comparisons],
    "fold_constants",
)

##########
#
# Now let's look at some rules that apply to nodes that are *not* all constants.
#
##########

# The first problem we're trying to solve here is that "(x + 1) + (y + 2)" or
# can be optimized to "(x + 3) + y" but we have no way given the rules
# above of discovering that. But here's what we can do.
#
# First, we can rewrite the tree so that (x + 1) + (y + 2) is in the more
# normal form of x + 1 + y + 2.

_associative_operator: Pattern = match_any(
    ast.Add, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Mult
)
_ops_match_right: Pattern = PredicatePattern(
    lambda b: isinstance(b.op, type(b.right.op))
)
_associate_to_left: Rule = pattern_rules(
    [
        (
            match_every(
                binop(op=_associative_operator, right=binop(op=_associative_operator)),
                _ops_match_right,
            ),
            lambda b: ast.BinOp(
                op=b.op,
                left=ast.BinOp(op=b.op, left=b.left, right=b.right.left),
                right=b.right.right,
            ),
        ),
        (
            binop(op=ast.Sub, right=binop(op=ast.Sub)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Sub(), right=b.right.left),
                op=ast.Add(),
                right=b.right.right,
            ),
        ),
        (
            binop(op=ast.Sub, right=binop(op=ast.Add)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Sub(), right=b.right.left),
                op=ast.Sub(),
                right=b.right.right,
            ),
        ),
        (
            binop(op=ast.Add, right=binop(op=ast.Sub)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Add(), right=b.right.left),
                op=ast.Sub(),
                right=b.right.right,
            ),
        ),
        (
            binop(op=ast.Div, right=binop(op=ast.Div)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Div(), right=b.right.left),
                op=ast.Mult(),
                right=b.right.right,
            ),
        ),
        (
            binop(op=ast.Div, right=binop(op=ast.Mult)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Div(), right=b.right.left),
                op=ast.Div(),
                right=b.right.right,
            ),
        ),
        (
            binop(op=ast.Mult, right=binop(op=ast.Div)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Mult(), right=b.right.left),
                op=ast.Div(),
                right=b.right.right,
            ),
        ),
    ],
    "associate_to_left",
)


#
# We now need a way to apply this rule everywhere, repeatedly, until it can
# no longer be applied. That is, we wish to reach a fixpoint of this
# transformation. This combinator does so relatively efficiently:

_fix_associative_ops = many(_some_top_down(at_least_once(_associate_to_left)))
# This always succeeds and reaches a fixpoint.

# We have a rule that turns "1 < 2 < 3" into "True and True", which means that
# a straightforward "run the rule once on every node, leaves to root" does not
# necessarily reach a fixpoint. However, running the rule *many* times on
# each node, *until it fails*, does produce a fixpoint.

_fold_all_constants: Rule = _bottom_up(many(_fold_constants), "fold_all_constants")

# Python parses "-1" as UnaryOp(USub, Num(1)). We need to fold that to Num(-1)
# so that we can fold expressions like (-2)*(3). But we should then turn that
# Num(-6) back into UnaryOp(USub, Num(6)) when we're done.

_fix_unary_minus = PatternRule(
    negative_num, lambda n: ast.UnaryOp(op=ast.USub(), operand=ast.Num(-n.n))
)

_fix_all: Rule = _bottom_up(once(_fix_unary_minus))

_rules = Compose(_fold_all_constants, _fix_all)


def fold(node: ast.AST) -> ast.AST:
    return _rules(node).expect_success()
