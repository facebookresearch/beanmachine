#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pattern matching for ASTs"""
import ast
from platform import python_version
from typing import Any, Dict

from beanmachine.ppl.compiler.patterns import (
    anyPattern as _any,
    match_any,
    match_every,
    negate,
    Pattern,
    PredicatePattern,
    type_and_attributes,
)
from beanmachine.ppl.compiler.rules import RuleDomain

# To support different Python versions correctly, in particular changes from 3.8 to 3.9,
# some functionality defined in this module needs to be version dependent.

_python_version = [int(i) for i in python_version().split(".")[:2]]
_python_3_9_or_later = _python_version >= [3, 9]

# Assertions about changes across versions that we address in this module

if _python_3_9_or_later:
    dummy_value = ast.Constant(1)
    assert ast.Index(dummy_value) == dummy_value
else:
    dummy_value = ast.Constant(1)
    assert ast.Index(dummy_value) != dummy_value


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


def arguments(
    args: Pattern = _any,
    vararg: Pattern = _any,
    kwonlyargs: Pattern = _any,
    kw_defaults: Pattern = _any,
    kwarg: Pattern = _any,
    defaults: Pattern = _any,
) -> Pattern:
    return type_and_attributes(
        ast.arguments,
        {
            "args": args,
            "vararg": vararg,
            "kwonlyargs": kwonlyargs,
            "kw_defaults": kw_defaults,
            "kwarg": kwarg,
            "defaults": defaults,
        },
    )


def ast_assert(expr: Pattern = _any, msg: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Assert, {"expr": expr, "msg": msg})


def assign(targets: Pattern = _any, value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Assign, {"targets": targets, "value": value})


def aug_assign(
    target: Pattern = _any, op: Pattern = _any, value: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.AugAssign, {"target": target, "op": op, "value": value}
    )


# TODO: what should we do about AnnAssign?


def starred(value: Pattern = _any, ctx: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Starred, {"value": value, "ctx": ctx})


def attribute(
    value: Pattern = _any, attr: Pattern = _any, ctx: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Attribute, {"value": value, "attr": attr, "ctx": ctx}
    )


def binop(op: Pattern = _any, left: Pattern = _any, right: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.BinOp, {"op": op, "left": left, "right": right})


def boolop(op: Pattern = _any, values: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.BoolOp, {"op": op, "values": values})


def call(
    func: Pattern = _any, args: Pattern = _any, keywords: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Call, {"func": func, "args": args, "keywords": keywords}
    )


def call_to(id: Pattern = _any, args: Pattern = _any) -> Pattern:
    return call(func=match_any(attribute(attr=id), name(id=id)), args=args)


def id_from_call(c: ast.Call) -> str:
    f = c.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    raise ValueError("Unexpected argument to id_from_call")


def compare(left: Pattern = _any, ops: Pattern = _any, comparators: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": ops, "comparators": comparators}
    )


def binary_compare(op: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": _any, "ops": [op], "comparators": [_any]}
    )


def equal(left: Pattern = _any, right: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": [ast.Eq], "comparators": [right]}
    )


def not_equal(left: Pattern = _any, right: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": [ast.NotEq], "comparators": [right]}
    )


def greater_than(left: Pattern = _any, right: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": [ast.Gt], "comparators": [right]}
    )


def greater_than_equal(left: Pattern = _any, right: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": [ast.GtE], "comparators": [right]}
    )


def less_than(left: Pattern = _any, right: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": [ast.Lt], "comparators": [right]}
    )


def less_than_equal(left: Pattern = _any, right: Pattern = _any):
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": [ast.LtE], "comparators": [right]}
    )


def expr(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Expr, {"value": value})


def ast_while(
    test: Pattern = _any, body: Pattern = _any, orelse: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.While, {"test": test, "body": body, "orelse": orelse}
    )


def ast_generator(elt: Pattern = _any, generators: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.GeneratorExp, {"elt": elt, "generators": generators})


def ast_listComp(elt: Pattern = _any, generators: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.ListComp, {"elt": elt, "generators": generators})


def ast_setComp(elt: Pattern = _any, generators: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.SetComp, {"elt": elt, "generators": generators})


def ast_dictComp(
    key: Pattern = _any, value: Pattern = _any, generators: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.DictComp, {"key": key, "value": value, "generators": generators}
    )


def ast_boolop(op: Pattern = _any, values: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.BoolOp, {"op": op, "values": values})


def ast_compare(
    left: Pattern = _any, ops: Pattern = _any, comparators: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Compare, {"left": left, "ops": ops, "comparators": comparators}
    )


def ast_for(
    target: Pattern = _any,
    iter: Pattern = _any,
    body: Pattern = _any,
    orelse: Pattern = _any,
) -> Pattern:
    return type_and_attributes(
        ast.For, {"target": target, "iter": iter, "body": body, "orelse": orelse}
    )


def function_def(
    name: Pattern = _any,
    args: Pattern = _any,
    body: Pattern = _any,
    decorator_list: Pattern = _any,
    returns: Pattern = _any,
) -> Pattern:
    return type_and_attributes(
        ast.FunctionDef,
        {
            "name": name,
            "args": args,
            "body": body,
            "decorator_list": decorator_list,
            "returns": returns,
        },
    )


def if_exp(
    test: Pattern = _any, body: Pattern = _any, orelse: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.IfExp, {"test": test, "body": body, "orelse": orelse}
    )


def if_statement(
    test: Pattern = _any, body: Pattern = _any, orelse: Pattern = _any
) -> Pattern:
    return type_and_attributes(ast.If, {"test": test, "body": body, "orelse": orelse})


# Note: The following pattern definition is valid only for Python
# versions less than 3.9. As a result, it is followed by a
# version-dependent redefinition


def _index(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Index, {"value": value})


def index(value: Pattern = _any):
    if _python_3_9_or_later:
        return match_every(value, negate(slice_pattern()))
    else:
        return _index(value=value)


# The following definition should not be necessary in 3.9
# since ast.Index should be identity in this version. It is
# nevertheless included for clarity.


def ast_index(value, **other):
    if _python_3_9_or_later:
        return value
    else:
        return ast.Index(value=value, **other)


def get_value(slice_field):
    if _python_3_9_or_later:
        return slice_field
    else:
        return slice_field.value


def slice_pattern(
    lower: Pattern = _any, upper: Pattern = _any, step: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Slice, {"lower": lower, "upper": upper, "step": step}
    )


def keyword(arg: Pattern = _any, value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.keyword, {"arg": arg, "value": value})


def ast_list(elts: Pattern = _any, ctx: Pattern = _any, ast_op=ast.List) -> Pattern:
    return type_and_attributes(ast_op, {"elts": elts, "ctx": ctx})


def ast_luple(elts: Pattern = _any, ctx: Pattern = _any) -> Pattern:
    return match_any(
        type_and_attributes(ast.List, {"elts": elts, "ctx": ctx}),
        type_and_attributes(ast.Tuple, {"elts": elts, "ctx": ctx}),
    )


def ast_dict(keys: Pattern = _any, values: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Dict, {"keys": keys, "values": values})


def module(body: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Module, {"body": body})


def name(id: Pattern = _any, ctx: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Name, {"id": id, "ctx": ctx})


def name_constant(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.NameConstant, {"value": value})


def num(n: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Num, {"n": n})


def ast_str(s: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Str, {"s": s})


def subscript(
    value: Pattern = _any, slice: Pattern = _any, ctx: Pattern = _any
) -> Pattern:
    return type_and_attributes(
        ast.Subscript, {"value": value, "slice": slice, "ctx": ctx}
    )


def ast_return(value: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.Return, {"value": value})


def ast_if(
    test: Pattern = _any, body: Pattern = _any, orelse: Pattern = _any
) -> Pattern:
    return type_and_attributes(ast.If, {"test": test, "body": body, "orelse": orelse})


def unaryop(op: Pattern = _any, operand: Pattern = _any) -> Pattern:
    return type_and_attributes(ast.UnaryOp, {"op": op, "operand": operand})


def unarysub(operand: Pattern = _any) -> Pattern:
    return unaryop(op=ast.USub, operand=operand)


zero: Pattern = match_any(num(0), num(0.0))

number_constant: Pattern = ast.Num

non_zero_num: Pattern = match_every(number_constant, negate(zero))

negative_num: Pattern = match_every(
    number_constant, PredicatePattern(lambda n: n.n < 0)
)

ast_true: Pattern = name_constant(True)

ast_false: Pattern = name_constant(False)

constant_bool: Pattern = match_any(ast_true, ast_false)

constant_literal: Pattern = match_any(number_constant, constant_bool)

any_list: Pattern = ast.List
