#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast

import astor
from beanmachine.ppl.utils.ast_patterns import (
    ast_domain,
    binop,
    constant_numeric,
    unaryop,
)
from beanmachine.ppl.utils.fold_constants import _fold_unary_op, fold
from beanmachine.ppl.utils.rules import (
    Compose,
    FirstMatch as first,
    PatternRule,
    TryMany as many,
    TryOnce as once,
)
from beanmachine.ppl.utils.single_assignment import single_assignment


# TODO: Remove the sample annotations
# TODO: Memoize each sample method
# TODO: Detect unsupported operators
# TODO: Detect unsupported control flow
# TODO: Would be helpful if we could track original source code locations.
# TODO: Transform y**x into exp(x*log(y))

_top_down = ast_domain.top_down
_bottom_up = ast_domain.bottom_up


_eliminate_subtraction = PatternRule(
    binop(op=ast.Sub),
    lambda b: ast.BinOp(
        left=b.left, op=ast.Add(), right=ast.UnaryOp(op=ast.USub(), operand=b.right)
    ),
)

_fix_usub_usub = PatternRule(
    unaryop(op=ast.USub, operand=unaryop(op=ast.USub)), lambda u: u.operand.operand
)

_fold_usub_const = PatternRule(
    unaryop(op=ast.USub, operand=constant_numeric), _fold_unary_op
)

_rules = Compose(
    _top_down(once(_eliminate_subtraction)),
    _bottom_up(many(first([_fix_usub_usub, _fold_usub_const]))),
)


def to_python(source: str) -> str:
    a: ast.AST = ast.parse(source)
    f: ast.AST = fold(a)
    # The AST has now had constants folded and associative
    # operators are nested to the left.
    es: ast.AST = _rules(f).expect_success()
    # The AST has now eliminated all subtractions; negative constants
    # are represented as constants, not as USubs
    sa: ast.AST = single_assignment(es)
    # Now we're in single assignment form.
    # TODO: Do the main transformation here
    # TODO: Fix negative constants back to standard form.
    p: str = astor.to_source(sa)
    return p
