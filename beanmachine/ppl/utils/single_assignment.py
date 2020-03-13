#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast

from beanmachine.ppl.utils.ast_patterns import ast_domain, ast_return, name
from beanmachine.ppl.utils.patterns import Pattern, negate
from beanmachine.ppl.utils.rules import ListEdit, PatternRule, Rule, TryMany as many


_some_top_down = ast_domain.some_top_down
_not_identifier: Pattern = negate(name())


class SingleAssignment:
    _count: int
    _rules: Rule

    def __init__(self) -> None:
        self._count = 0
        self._rules = many(_some_top_down(self._handle_return()))

    def _unique_id(self, prefix: str) -> str:
        self._count = self._count + 1
        return f"{prefix}{self._count}"

    def _handle_return(self) -> Rule:
        def _fix_return(r: ast.Return) -> ListEdit:
            id = self._unique_id("r")
            return ListEdit(
                [
                    ast.Assign(
                        targets=[ast.Name(id=id, ctx=ast.Store())], value=r.value
                    ),
                    ast.Return(value=ast.Name(id=id, ctx=ast.Load())),
                ]
            )

        return PatternRule(
            ast_return(value=_not_identifier), _fix_return, "handle_return"
        )

    def single_assignment(self, node: ast.AST) -> ast.AST:
        return self._rules(node).expect_success()


def single_assignment(node: ast.AST) -> ast.AST:
    s = SingleAssignment()
    return s.single_assignment(node)
