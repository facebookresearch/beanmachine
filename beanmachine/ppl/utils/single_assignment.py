#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
from typing import Callable

from beanmachine.ppl.utils.ast_patterns import (
    assign,
    ast_domain,
    ast_return,
    binop,
    match_any,
    name,
)
from beanmachine.ppl.utils.patterns import Pattern, negate
from beanmachine.ppl.utils.rules import (
    FirstMatch as first,
    ListEdit,
    PatternRule,
    Rule,
    TryMany as many,
    pattern_rules,
)


_some_top_down = ast_domain.some_top_down
_not_identifier: Pattern = negate(name())
_binops: Pattern = match_any(
    ast.Add,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Div,
    ast.LShift,
    ast.Mod,
    ast.Mult,
    ast.Pow,
    ast.RShift,
    ast.Sub,
)


class SingleAssignment:
    _count: int
    _rules: Rule

    def __init__(self) -> None:
        self._count = 0
        self._rules = many(
            _some_top_down(first([self._handle_return(), self._handle_assign()]))
        )

    def _unique_id(self, prefix: str) -> str:
        self._count = self._count + 1
        return f"{prefix}{self._count}"

    def _fix_it(
        self,
        prefix: str,
        value: Callable[[ast.AST], ast.AST],
        replace: Callable[[ast.AST, ast.AST], ast.AST],
    ) -> Callable[[ast.AST], ListEdit]:
        def _do_it(r: ast.AST) -> ListEdit:
            id = self._unique_id(prefix)
            return ListEdit(
                [
                    ast.Assign(
                        targets=[ast.Name(id=id, ctx=ast.Store())], value=value(r)
                    ),
                    replace(r, ast.Name(id=id, ctx=ast.Load())),
                ]
            )

        return _do_it

    def _handle_return(self) -> Rule:
        return PatternRule(
            ast_return(value=_not_identifier),
            self._fix_it("r", lambda r: r.value, lambda r, v: ast.Return(value=v)),
            "handle_return",
        )

    def _handle_assign(self) -> Rule:
        return pattern_rules(
            [
                (
                    assign(value=binop(left=_not_identifier, op=_binops)),
                    self._fix_it(
                        "a",
                        lambda a: a.value.left,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.BinOp(left=v, op=a.value.op, right=a.value.right),
                        ),
                    ),
                ),
                (
                    assign(value=binop(right=_not_identifier, op=_binops)),
                    self._fix_it(
                        "a",
                        lambda a: a.value.right,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.BinOp(left=a.value.left, op=a.value.op, right=v),
                        ),
                    ),
                ),
            ],
            "handle_assign",
        )

    def single_assignment(self, node: ast.AST) -> ast.AST:
        return self._rules(node).expect_success()


def single_assignment(node: ast.AST) -> ast.AST:
    s = SingleAssignment()
    return s.single_assignment(node)
