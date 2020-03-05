#!/usr/bin/env python3
"""Pattern matching for ASTs"""
from ast import Add, BinOp, Expr, Module, NameConstant, Num, Str

from beanmachine.ppl.utils.patterns import (
    Pattern,
    anyPattern as _any,
    type_and_attributes,
)


add: Pattern = Add


def binop(op: Pattern = _any, left: Pattern = _any, right: Pattern = _any) -> Pattern:
    return type_and_attributes(BinOp, [("op", op), ("left", left), ("right", right)])


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
