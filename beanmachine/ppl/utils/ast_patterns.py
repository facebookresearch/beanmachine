#!/usr/bin/env python3
"""Pattern matching for ASTs"""
import ast
from typing import Any, Dict

from beanmachine.ppl.utils.patterns import (
    Pattern,
    anyPattern as _any,
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


add: Pattern = ast.Add

ast_is: Pattern = ast.Is

load: Pattern = ast.Load

ast_pass: Pattern = ast.Pass


def binop(op: Pattern = _any, left: Pattern = _any, right: Pattern = _any) -> Pattern:
    return type_and_attributes(
        ast.BinOp, [("op", op), ("left", left), ("right", right)]
    )


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
