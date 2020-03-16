#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast

from beanmachine.ppl.utils.ast_patterns import (
    ast_domain,
    boolop,
    constant_falsy,
    constant_truthy,
    if_statement,
)
from beanmachine.ppl.utils.fold_constants import fix_unary_minus, fold_unary_minus
from beanmachine.ppl.utils.patterns import HeadTail, anyPattern as _any
from beanmachine.ppl.utils.rules import (
    AllOf as _all,
    ListEdit,
    Rule,
    TryMany as many,
    TryOnce as once,
    pattern_rules,
)


_top_down = ast_domain.top_down


##########
#
# These operations optimize expressions or statements that are not necessarily
# entirely constant, but have some constant element. For optimizations that
# deal only with constants, see fold_constants.py
#
##########


_optimize_logic: Rule = pattern_rules(
    [
        # (x and ...) becomes x if x is logically false.
        (boolop(op=ast.And, values=HeadTail(constant_falsy)), lambda b: b.values[0]),
        # (x and y) becomes y if x is logically true.
        (
            boolop(op=ast.And, values=HeadTail(constant_truthy, [_any])),
            lambda b: b.values[1],
        ),
        # (x and ...) becomes ... if x is logically true.
        (
            boolop(op=ast.And, values=HeadTail(constant_truthy)),
            lambda b: ast.BoolOp(op=b.op, values=b.values[1:]),
        ),
        # (x or ...) becomes x if x is logically true.
        (boolop(op=ast.Or, values=HeadTail(constant_truthy)), lambda b: b.values[0]),
        # (x or y) becomes y if x is logically false.
        (
            boolop(op=ast.Or, values=HeadTail(constant_falsy, [_any])),
            lambda b: b.values[1],
        ),
        # (x or ...) becomes ... if x is logically false.
        (
            boolop(op=ast.Or, values=HeadTail(constant_falsy)),
            lambda b: ast.BoolOp(op=b.op, values=b.values[1:]),
        ),
    ],
    "optimize_logic",
)

_optimize_if: Rule = pattern_rules(
    [
        (if_statement(test=constant_truthy), lambda s: ListEdit(s.body)),
        # The orelse is [] if the else clause is missing, so this is fine.
        (if_statement(test=constant_falsy), lambda s: ListEdit(s.orelse)),
    ]
)

_rules = _all([_top_down(many(_optimize_logic)), _top_down(once(_optimize_if))])


def optimize(node: ast.AST) -> ast.AST:
    n1 = fold_unary_minus(node)
    n2 = _rules(n1).expect_success()
    n3 = fix_unary_minus(n2)
    return n3
