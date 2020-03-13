#!/usr/bin/env python3
"""Pattern matching for ASTs"""
import ast
from typing import Any, Dict

from beanmachine.ppl.utils.patterns import (
    ListAll as list_all,
    MatchResult,
    Pattern,
    PatternBase,
    PredicatePattern,
    anyPattern as _any,
    match,
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


def assign(targets: Pattern = _any, value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Assign, [("targets", targets), ("value", value)])


def attribute(
    value: Pattern = _any, attr: Pattern = _any, ctx: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Attribute, [("value", value), ("attr", attr), ("ctx", ctx)]
    )


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


def if_exp(
    test: Pattern = _any, body: Pattern = _any, orelse: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.IfExp, [("test", test), ("body", body), ("orelse", orelse)]
    )


def ast_list(elts: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.List, [("elts", elts)])


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


zero: Pattern = match_any(num(0), num(0.0))

number_constant: Pattern = ast.Num

non_zero_num: Pattern = match_every(number_constant, negate(zero))

negative_num: Pattern = match_every(
    number_constant, PredicatePattern(lambda n: n.n < 0)
)

ast_true: Pattern = name_constant(True)

ast_false: Pattern = name_constant(False)

bool_constant: Pattern = match_any(ast_true, ast_false)

bool_constant: Pattern = match_any(ast_true, ast_false)

any_constant: Pattern = match_any(number_constant, bool_constant)


constant_list: PatternBase


class ConstantList(PatternBase):
    """A recursively-defined pattern which matches a list expression containing only
    numeric literals, or lists of numeric literals, and so on."""

    # Note that the empty list does match; that's by design.

    def match(self, test: Any) -> MatchResult:
        return match(
            ast_list(elts=list_all(match_any(number_constant, constant_list))), test
        )

    def _to_str(self, test: str) -> str:
        return f"{test} is a constant list"


constant_list = ConstantList()

tensor_name_str: Pattern = "tensor"
tensor_name: Pattern = name(id=tensor_name_str)
# TODO: Matches "tensor" and "foo.tensor"
# TODO: Do we need to specifically match just torch? What if there is an alias?
tensor_ctor: Pattern = match_any(attribute(attr=tensor_name_str), tensor_name)

# Recognizes tensor(1)
constant_tensor_1: Pattern = call(func=tensor_ctor, args=[number_constant])

# Recognizes tensor(1), tensor([]), tensor([1, 2]), tensor([[1, 2], [3, 4]]) and so on
constant_tensor_any: Pattern = call(
    func=tensor_ctor, args=[match_any(number_constant, constant_list)]
)
