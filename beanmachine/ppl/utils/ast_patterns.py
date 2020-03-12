#!/usr/bin/env python3
"""Pattern matching for ASTs"""
from ast import (
    AST,
    Add,
    BinOp,
    Compare,
    Expr,
    Is,
    Load,
    Module,
    NameConstant,
    Num,
    Str,
    iter_fields,
)
from typing import Any, Dict

from beanmachine.ppl.utils.patterns import (
    Pattern,
    anyPattern as _any,
    type_and_attributes,
)
from beanmachine.ppl.utils.rules import RuleDomain


def _get_children(node: Any) -> Dict[str, Any]:
    if isinstance(node, AST):
        return dict(iter_fields(node))
    return {}


def _construct(typ: type, children: Dict[str, AST]) -> AST:
    return typ(**children)


ast_domain = RuleDomain(_get_children, _construct)


add: Pattern = Add

ast_is: Pattern = Is

load: Pattern = Load


def binop(op: Pattern = _any, left: Pattern = _any, right: Pattern = _any) -> Pattern:
    return type_and_attributes(BinOp, [("op", op), ("left", left), ("right", right)])


def compare(left: Pattern = _any, ops: Pattern = _any, comparators: Pattern = _any):
    return type_and_attributes(
        Compare, [("left", left), ("ops", ops), ("comparators", comparators)]
    )


def expr(value: Pattern = _any) -> Pattern:
    return type_and_attributes(Expr, [("value", value)])


def module(body: Pattern = _any) -> Pattern:
    return type_and_attributes(Module, [("body", body)])


def name_constant(value: Pattern = _any) -> Pattern:
    return type_and_attributes(NameConstant, [("value", value)])


def num(n: Pattern = _any) -> Pattern:
    return type_and_attributes(Num, [("n", n)])


def ast_str(s: Pattern = _any) -> Pattern:
    return type_and_attributes(Str, [("s", s)])
