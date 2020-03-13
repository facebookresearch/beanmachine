#!/usr/bin/env python3
"""Pattern matching for ASTs"""
import ast
from typing import Any, Dict

from beanmachine.ppl.utils.patterns import (
    Pattern,
    PredicatePattern,
    anyPattern as _any,
    match_any,
    match_every,
    negate,
    type_and_attributes,
)
from beanmachine.ppl.utils.rules import RuleDomain


def _get_children(node: Any) -> Dict[str, Any]:
    if isinstance(node, ast.AST):
        return dict(ast.iter_fields(node))
    return {}


def _construct(typ: type, children: Dict[str, ast.AST]) -> ast.AST:
    return typ(**children)


ast_domain = RuleDomain(_get_children, _construct)


ast_and: Pattern = ast.And

add: Pattern = ast.Add

bit_and: Pattern = ast.BitAnd

bit_or: Pattern = ast.BitOr

bit_xor: Pattern = ast.BitXor

div: Pattern = ast.Div

eq: Pattern = ast.Eq

gt: Pattern = ast.Gt

gte: Pattern = ast.GtE

invert: Pattern = ast.Invert

ast_is: Pattern = ast.Is

ast_is_not: Pattern = ast.IsNot

load: Pattern = ast.Load

lshift: Pattern = ast.LShift

lt: Pattern = ast.Lt

lte: Pattern = ast.LtE

mod: Pattern = ast.Mod

mult: Pattern = ast.Mult

not_eq: Pattern = ast.NotEq

ast_or: Pattern = ast.Or

ast_pass: Pattern = ast.Pass

ast_pow: Pattern = ast.Pow

rshift: Pattern = ast.RShift

sub: Pattern = ast.Sub

uadd: Pattern = ast.UAdd

usub: Pattern = ast.USub


def binop(op: Pattern = _any, left: Pattern = _any, right: Pattern = _any) -> Pattern:
    return type_and_attributes(
        ast.BinOp, [("op", op), ("left", left), ("right", right)]
    )


def boolop(op: Pattern = _any, values: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.BoolOp, [("op", op), ("values", values)])


def call(
    func: Pattern = _any, args: Pattern = _any, keywords: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Call, [("func", func), ("args", args), ("keywords", keywords)]
    )


def compare(left: Pattern = _any, ops: Pattern = _any, comparators: Pattern = _any):
    return type_and_attributes(
        ast.Compare, [("left", left), ("ops", ops), ("comparators", comparators)]
    )


def expr(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Expr, [("value", value)])


def function_def(
    name: Pattern = _any,
    args: Pattern = _any,
    body: Pattern = _any,
    decorator_list: Pattern = _any,
    returns: Pattern = _any,
) -> Pattern:
    return type_and_attributes(
        ast.FunctionDef,
        [
            ("name", name),
            ("args", args),
            ("body", body),
            ("decorator_list", decorator_list),
            ("returns", returns),
        ],
    )


def if_exp(test: Pattern = _any, body: Pattern = _any, orelse: Pattern = _any):
    return type_and_attributes(
        ast.IfExp, [("test", test), ("body", body), ("orelse", orelse)]
    )


def module(body: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Module, [("body", body)])


def name(id: Pattern = _any, ctx: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Name, [("id", id), ("ctx", ctx)])


def name_constant(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.NameConstant, [("value", value)])


def num(n: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Num, [("n", n)])


def ast_str(s: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Str, [("s", s)])


def ast_return(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Return, [("value", value)])


def unaryop(op: Pattern = _any, operand: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.UnaryOp, [("op", op), ("operand", operand)])


zero = match_any(num(0), num(0.0))

non_zero_num = match_every(num(), negate(zero))

negative_num = match_every(num(), PredicatePattern(lambda n: n.n < 0))

ast_true = name_constant(True)

ast_false = name_constant(False)

bool_constant = match_any(ast_true, ast_false)

any_constant = match_any(num(), name_constant())
