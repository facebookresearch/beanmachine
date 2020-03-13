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
    if_exp,
    negative_num,
    non_zero_num,
    num,
    unaryop,
)
from beanmachine.ppl.utils.patterns import ListAll
from beanmachine.ppl.utils.rules import (
    Compose,
    FirstMatch as first,
    PatternRule,
    TryOnce as once,
    pattern_rules,
)


_bottom_up = ast_domain.bottom_up

# TODO: Comparison operators are strange in Python; we can do a fold on them
# TODO: if we're clever.
# TODO: Fold operations on constant tensors.
# TODO: Fold matmul?

# These operations fold expressions that are entirely constant. Folding expressions
# such as "x and true" to "x" will be a different pass.

# TODO: For the associative operations, turn "nonconst + const + const" into
# TODO: "const + const + nonconst" first, so that the constants can be folded.

_fold_arithmetic = pattern_rules(
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

_fold_logic = pattern_rules(
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

_fold_conditional = pattern_rules(
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

_fold_constants = first(
    [_fold_arithmetic, _fold_logic, _fold_conditional], "fold_constants"
)

_fold_all_constants = _bottom_up(once(_fold_constants), "fold_all_constants")

# Python parses "-1" as UnaryOp(USub, Num(1)). We need to fold that to Num(-1)
# so that we can fold expressions like (-2)*(3). But we should then turn that
# Num(-6) back into UnaryOp(USub, Num(6)) when we're done.

_fix_unary_minus = PatternRule(
    negative_num, lambda n: ast.UnaryOp(op=ast.USub(), operand=ast.Num(-n.n))
)

_fix_all = _bottom_up(once(_fix_unary_minus))

_rules = Compose(_fold_all_constants, _fix_all)


def fold(node: ast.AST) -> ast.AST:
    return _rules(node).expect_success()
