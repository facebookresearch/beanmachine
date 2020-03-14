#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
from typing import List

import astor
from beanmachine.ppl.utils.ast_patterns import (
    assign,
    ast_domain,
    binop,
    bool_constant,
    call_to,
    constant_numeric,
    constant_tensor_any,
    unaryop,
)
from beanmachine.ppl.utils.fold_constants import _fold_unary_op, fold
from beanmachine.ppl.utils.rules import (
    AllOf as all_of,
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
# TODO: Collapse adds and multiplies

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

_bmg = ast.Name(id="bmg", ctx=ast.Load())


def _make_bmg_call(name: str, args: List[ast.AST]) -> ast.AST:
    return ast.Call(func=ast.Attribute(value=_bmg, attr=name), args=args, keywords=[])


_add_boolean = PatternRule(
    assign(value=bool_constant),
    lambda a: ast.Assign(a.targets, _make_bmg_call("add_boolean", [a.value])),
)

_add_real = PatternRule(
    assign(value=ast.Num),
    lambda a: ast.Assign(a.targets, _make_bmg_call("add_real", [a.value])),
)

_add_tensor = PatternRule(
    assign(value=constant_tensor_any),
    lambda a: ast.Assign(a.targets, _make_bmg_call("add_tensor", [a.value])),
)

_add_negate = PatternRule(
    assign(value=unaryop(op=ast.USub)),
    lambda a: ast.Assign(a.targets, _make_bmg_call("add_negate", [a.value.operand])),
)

_add_addition = PatternRule(
    assign(value=binop(op=ast.Add)),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("add_addition", [a.value.left, a.value.right])
    ),
)

_add_multiplication = PatternRule(
    assign(value=binop(op=ast.Mult)),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("add_multiplication", [a.value.left, a.value.right])
    ),
)

_add_exp = PatternRule(
    assign(value=call_to(id="exp")),
    lambda a: ast.Assign(a.targets, _make_bmg_call("add_exp", [a.value.args[0]])),
)

_add_bernoulli = PatternRule(
    assign(value=call_to(id="Bernoulli")),
    lambda a: ast.Assign(a.targets, _make_bmg_call("add_bernoulli", [a.value.args[0]])),
)

# TODO: add_to_real
# TODO: add_sample
# TODO: add_observation

_to_bmg = first(
    [
        _add_boolean,
        _add_real,
        _add_tensor,
        _add_negate,
        _add_addition,
        _add_multiplication,
        _add_exp,
        _add_bernoulli,
    ]
)


_rules = all_of(
    [
        _top_down(once(_eliminate_subtraction)),
        _bottom_up(many(first([_fix_usub_usub, _fold_usub_const]))),
    ]
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
    bmg: ast.AST = _top_down(once(_to_bmg))(sa).expect_success()
    # TODO: Fix negative constants back to standard form.
    p: str = astor.to_source(bmg)
    return p
