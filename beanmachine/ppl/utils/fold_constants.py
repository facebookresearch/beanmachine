#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
import math
from typing import Any, Callable, Dict

import torch
from beanmachine.ppl.utils.ast_patterns import (
    ast_domain,
    ast_to_constant_value,
    binop,
    boolop,
    call_to,
    compare,
    constant_literal,
    constant_numeric,
    constant_tensor_any,
    constant_value_to_ast,
    if_exp,
    match_any,
    negative_num,
    unaryop,
)
from beanmachine.ppl.utils.patterns import (
    ListAll,
    Pattern,
    PredicatePattern,
    match_every,
    negate,
)
from beanmachine.ppl.utils.rules import (
    Compose,
    FirstMatch as first,
    PatternRule,
    Rule,
    SomeOf,
    TryMany as many,
    TryOnce as once,
    at_least_once,
    ignore_div_zero,
    ignore_runtime_error,
    ignore_value_error,
    pattern_rules,
)


_all = ast_domain.all_children
_bottom_up = ast_domain.bottom_up
_some_bottom_up = ast_domain.some_bottom_up
_some_top_down = ast_domain.some_top_down
_some = ast_domain.some_children

# TODO: Fold matmul?


##########
#
# These operations fold expressions that are entirely constant. Folding expressions
# such as "x and true" to "x" will be a different pass.
# TODO: Optimization pass
#
##########


_unary_folders: Dict[type, Callable[[Any], Any]] = {
    ast.USub: lambda u: -u,
    ast.UAdd: lambda u: +u,
    ast.Invert: lambda u: ~u,
    ast.Not: lambda u: not u,
}

_binary_folders: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda left, right: left + right,
    ast.BitAnd: lambda left, right: left & right,
    ast.BitOr: lambda left, right: left | right,
    ast.BitXor: lambda left, right: left ^ right,
    ast.Div: lambda left, right: left / right,
    ast.LShift: lambda left, right: left << right,
    ast.Mod: lambda left, right: left % right,
    ast.Mult: lambda left, right: left * right,
    ast.Pow: lambda left, right: left ** right,
    ast.RShift: lambda left, right: left >> right,
    ast.Sub: lambda left, right: left - right,
}


def _fold_unary_op(u: ast.UnaryOp) -> ast.AST:
    return constant_value_to_ast(
        _unary_folders[type(u.op)](ast_to_constant_value(u.operand))
    )


_fold_unary: Rule = PatternRule(
    unaryop(
        op=match_any(ast.USub, ast.UAdd, ast.Invert, ast.Not), operand=constant_numeric
    ),
    _fold_unary_op,
)


def _fold_binary_op(b: ast.BinOp) -> ast.AST:
    return constant_value_to_ast(
        _binary_folders[type(b.op)](
            ast_to_constant_value(b.left), ast_to_constant_value(b.right)
        )
    )


_fold_binary_1: Rule = PatternRule(
    binop(
        op=match_any(
            ast.Add,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.LShift,
            ast.Mult,
            ast.Pow,
            ast.RShift,
            ast.Sub,
        ),
        left=constant_numeric,
        right=constant_numeric,
    ),
    _fold_binary_op,
    "fold_binary_1",
)


_fold_binary_2: Rule = ignore_div_zero(
    PatternRule(
        binop(
            op=match_any(ast.Div, ast.Mod),
            left=constant_numeric,
            right=constant_numeric,
        ),
        _fold_binary_op,
        "fold_binary_2",
    )
)

_fold_arithmetic = first([_fold_unary, _fold_binary_1, _fold_binary_2])


def _fold_and(b: ast.BoolOp) -> ast.AST:
    current = ast_to_constant_value(b.values[0])
    for v in b.values[1:]:
        current = current and ast_to_constant_value(v)
    return constant_value_to_ast(current)


def _fold_or(b: ast.BoolOp) -> ast.AST:
    current = ast_to_constant_value(b.values[0])
    for v in b.values[1:]:
        current = current or ast_to_constant_value(v)
    return constant_value_to_ast(current)


_fold_logic: Rule = pattern_rules(
    [
        (boolop(op=ast.And, values=ListAll(constant_numeric)), _fold_and),
        (boolop(op=ast.Or, values=ListAll(constant_numeric)), _fold_or),
    ],
    "fold_logic",
)

_fold_conditional: Rule = PatternRule(
    if_exp(test=constant_numeric, body=constant_numeric, orelse=constant_numeric),
    lambda i: constant_value_to_ast(
        ast_to_constant_value(i.body)
        if ast_to_constant_value(i.test)
        else ast_to_constant_value(i.orelse)
    ),
    "fold_conditional",
)

_comparison_folders: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: lambda left, right: left == right,
    ast.Gt: lambda left, right: left > right,
    ast.GtE: lambda left, right: left >= right,
    ast.Is: lambda left, right: left is right,
    ast.IsNot: lambda left, right: left is not right,
    ast.Lt: lambda left, right: left < right,
    ast.LtE: lambda left, right: left <= right,
    ast.NotEq: lambda left, right: left != right,
}


def _fold_comparison_op(c: ast.Compare) -> ast.AST:
    current_left = c.left
    result = True
    for op, current_right in zip(c.ops, c.comparators):
        result = result and _comparison_folders[type(op)](
            ast_to_constant_value(current_left), ast_to_constant_value(current_right)
        )
        current_left = current_right
    return constant_value_to_ast(result)


_fold_comparison: Rule = PatternRule(
    compare(
        left=constant_numeric,
        ops=ListAll(
            match_any(
                ast.Eq, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.Lt, ast.LtE, ast.NotEq
            )
        ),
        comparators=ListAll(constant_numeric),
    ),
    _fold_comparison_op,
    "fold_comparison",
)


_fold_tensor_log: Rule = PatternRule(
    call_to(id="log", args=[constant_tensor_any]),
    lambda c: constant_value_to_ast(torch.log(ast_to_constant_value(c.args[0]))),
    "fold_tensor_log",
)

_fold_log: Rule = ignore_value_error(
    PatternRule(
        call_to(id="log", args=[constant_literal]),
        lambda c: constant_value_to_ast(math.log(ast_to_constant_value(c.args[0]))),
        "fold_log",
    )
)

_fold_pure: Rule = first([_fold_tensor_log, _fold_log], "fold_pure")

_fold_constants = ignore_runtime_error(
    first(
        [
            _fold_arithmetic,
            _fold_logic,
            _fold_conditional,
            _fold_comparison,
            _fold_pure,
        ],
        "fold_constants",
    )
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
            # x op (y op z) => x op y op z
            lambda b: ast.BinOp(
                op=b.op,
                left=ast.BinOp(op=b.op, left=b.left, right=b.right.left),
                right=b.right.right,
            ),
        ),
        (
            # x - (y - z) => x - y + z
            binop(op=ast.Sub, right=binop(op=ast.Sub)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Sub(), right=b.right.left),
                op=ast.Add(),
                right=b.right.right,
            ),
        ),
        (
            # x - (y + z) => x - y - z
            binop(op=ast.Sub, right=binop(op=ast.Add)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Sub(), right=b.right.left),
                op=ast.Sub(),
                right=b.right.right,
            ),
        ),
        (
            # x + (y - z) => x + y - z
            binop(op=ast.Add, right=binop(op=ast.Sub)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Add(), right=b.right.left),
                op=ast.Sub(),
                right=b.right.right,
            ),
        ),
        (
            # x / (y / z) => x / y * z
            binop(op=ast.Div, right=binop(op=ast.Div)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Div(), right=b.right.left),
                op=ast.Mult(),
                right=b.right.right,
            ),
        ),
        (
            # x / (y * z) => x / y / z
            binop(op=ast.Div, right=binop(op=ast.Mult)),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left, op=ast.Div(), right=b.right.left),
                op=ast.Div(),
                right=b.right.right,
            ),
        ),
        (
            # x * (y / z) => x * y / z
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
# transformation. However, we also still want to know if no progress is being
# made. We can meet both of these needs like this:

_fix_associative_ops = _some_top_down(at_least_once(_associate_to_left))
# This fails if no progress is made. many(_fix_associative_ops) reaches a fixpoint
# and always succeeds.

# Second, we can make two new rules, where C is "constant" and N is "not constant":
#
# (X + C1) + C2 => X + (C1 + C2)
# (X + N) + C   => (X + C) + N
#
# and then we have a fold opportunity on the new children. By repeatedly applying
# these rules to an expression in its normal form we gradually eliminate all the
# folding possibilities and reach a fixpoint.
#
# That is, in our example we will go from x + 1 + y + 2 to:
#
# x + 1 + 2 + y
# x + (1 + 2) + y
# x + 3 + y
#
# and now we can't go further.
#
# Plainly we wish to apply this rule from top to bottom, since the second operation
# moves the constant deeper into the tree.

_ops_match_left: Pattern = PredicatePattern(lambda b: isinstance(b.op, type(b.left.op)))

_not_a_constant_number: Pattern = negate(constant_numeric)

_const_to_right: Rule = pattern_rules(
    [
        (
            match_every(
                binop(
                    op=_associative_operator,
                    left=binop(op=_associative_operator, right=constant_numeric),
                    right=constant_numeric,
                ),
                _ops_match_left,
            ),
            lambda b: ast.BinOp(
                op=b.op,
                left=b.left.left,
                right=ast.BinOp(op=b.op, left=b.left.right, right=b.right),
            ),
        ),
        (
            # x - c1 + c2 => x + (c2 - c1)
            binop(
                left=binop(op=ast.Sub, right=constant_numeric),
                op=ast.Add,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=b.left.left,
                op=ast.Add(),
                right=ast.BinOp(left=b.right, op=ast.Sub(), right=b.left.right),
            ),
        ),
        (
            # x - c1 - c2 => x - (c1 + c2)
            binop(
                left=binop(op=ast.Sub, right=constant_numeric),
                op=ast.Sub,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=b.left.left,
                op=ast.Sub(),
                right=ast.BinOp(left=b.left.right, op=ast.Add(), right=b.right),
            ),
        ),
        (
            # x + c1 - c2 => x + (c1 - c2)
            binop(
                left=binop(op=ast.Add, right=constant_numeric),
                op=ast.Sub,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=b.left.left,
                op=ast.Add(),
                right=ast.BinOp(left=b.left.right, op=ast.Sub(), right=b.right),
            ),
        ),
        (
            # x / c1 * c2 => x * (c2 / c1)
            binop(
                left=binop(op=ast.Div, right=constant_numeric),
                op=ast.Mult,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=b.left.left,
                op=ast.Mult(),
                right=ast.BinOp(left=b.right, op=ast.Div(), right=b.left.right),
            ),
        ),
        (
            # x / c1 / c2 => x / (c1 * c2)
            binop(
                left=binop(op=ast.Div, right=constant_numeric),
                op=ast.Div,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=b.left.left,
                op=ast.Div(),
                right=ast.BinOp(left=b.left.right, op=ast.Mult(), right=b.right),
            ),
        ),
        (
            # x * c1 / c2 => x * (c1 / c2)
            binop(
                left=binop(op=ast.Mult, right=constant_numeric),
                op=ast.Div,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=b.left.left,
                op=ast.Mult(),
                right=ast.BinOp(left=b.left.right, op=ast.Div(), right=b.right),
            ),
        ),
    ],
    "const_to_right",
)
_const_to_left: Rule = pattern_rules(
    [
        (
            match_every(
                binop(
                    op=_associative_operator,
                    left=binop(op=_associative_operator, right=_not_a_constant_number),
                    right=constant_numeric,
                ),
                _ops_match_left,
            ),
            lambda b: ast.BinOp(
                op=b.op,
                left=ast.BinOp(op=b.op, left=b.left.left, right=b.right),
                right=b.left.right,
            ),
        ),
        (
            # x - n + c => x + c - n
            binop(
                left=binop(op=ast.Sub, right=_not_a_constant_number),
                op=ast.Add,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left.left, op=ast.Add(), right=b.right),
                op=ast.Sub(),
                right=b.left.right,
            ),
        ),
        (
            # x - n - c => x - c - n
            binop(
                left=binop(op=ast.Sub, right=_not_a_constant_number),
                op=ast.Sub,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left.left, op=ast.Sub(), right=b.right),
                op=ast.Sub(),
                right=b.left.right,
            ),
        ),
        (
            # x + n - c => x - c + n
            binop(
                left=binop(op=ast.Add, right=_not_a_constant_number),
                op=ast.Sub,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left.left, op=ast.Sub(), right=b.right),
                op=ast.Add(),
                right=b.left.right,
            ),
        ),
        (
            # x / n * c => x * c / n
            binop(
                left=binop(op=ast.Div, right=_not_a_constant_number),
                op=ast.Mult,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left.left, op=ast.Mult(), right=b.right),
                op=ast.Div(),
                right=b.left.right,
            ),
        ),
        (
            # x / n / c => x / c / n
            binop(
                left=binop(op=ast.Div, right=_not_a_constant_number),
                op=ast.Div,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left.left, op=ast.Div(), right=b.right),
                op=ast.Div(),
                right=b.left.right,
            ),
        ),
        (
            # x * n / c => x / c * n
            binop(
                left=binop(op=ast.Mult, right=_not_a_constant_number),
                op=ast.Div,
                right=constant_numeric,
            ),
            lambda b: ast.BinOp(
                left=ast.BinOp(left=b.left.left, op=ast.Div(), right=b.right),
                op=ast.Mult(),
                right=b.left.right,
            ),
        ),
    ],
    "const_to_left",
)

# We want to move a constant around if we can; if we do, then we want to
# try to apply a fold to the new children that were just created.
#
# We can easily get in a situation where applying this rule once makes
# it possible that it applies again, so we'll keep trying to do it until
# we can no longer.
#
# However, there is a subtle point to consider here. Suppose the program
# we are manipulating is wrong and will fail at runtime. We still need to
# be able to run the constant folding pass on it and preserve the program
# semantics, without going into an infinite loop.  Suppose for instance
# that we are in the situation "x + t1 + t2" where t1 and t2 are constant
# tensors that cannot be added. If const_to_right turns this into
# x + (t1 + t2) and the constant folding fails to apply, and we have "made
# progress" and do not fail, then we've made work for _fix_associative_ops
# to do; it will turn it back into x + t1 + t2, and now we're in a loop
# of two rules undoing each other. What we need to do is ensure that
# const_to_right only succeeds if the fold succeeds.
#
# By constrast, if we go from t1 + x + t2 to t1 + t2 + x, and then constant
# folding fails to apply, we're OK. We've still made progress and no other
# rule will undo that.

_move_constants: Rule = _some_top_down(
    at_least_once(
        first(
            [
                Compose(_const_to_right, _some(_fold_constants)),
                Compose(_const_to_left, once(_some(_fold_constants))),
            ]
        )
    )
)
# This fails if we can't move a constant.

# We have a rule that turns "1 < 2 < 3" into "True and True", which means that
# a straightforward "run the rule once on every node, leaves to root" does not
# necessarily reach a fixpoint. However, running the rule *many* times on
# each node, *until it fails*, does produce a fixpoint.

# All right; let's put it all together. To fold, we need to:
# (1) Canonicalize associative operators
# (2) Move constants to the left in associative operators
# (3) Fold constants
# Each one of these rules fails if it does not make progress,
# so the whole thing fails if it does not make progress.


_fold_all_constants: Rule = many(
    SomeOf(
        [
            _fix_associative_ops,
            _move_constants,
            _some_bottom_up(at_least_once(_fold_constants)),
        ],
        "fold_all_constants",
    )
)


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
