#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
from typing import Callable, List, Tuple

from beanmachine.ppl.utils.ast_patterns import (
    assign,
    ast_domain,
    ast_for,
    ast_list,
    ast_return,
    attribute,
    binop,
    call,
    expr,
    index,
    keyword,
    match,
    match_any,
    name,
    subscript,
    unaryop,
)
from beanmachine.ppl.utils.patterns import (
    ListAll,
    ListAny,
    Pattern,
    PatternBase,
    negate,
)
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
_list_not_identifier: PatternBase = ListAny(_not_identifier)
_list_all_identifiers: PatternBase = ListAll(name())
_not_identifier_keyword: Pattern = keyword(value=_not_identifier)
_not_identifier_keywords: PatternBase = ListAny(_not_identifier_keyword)

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

_unops: Pattern = match_any(ast.USub, ast.UAdd, ast.Invert, ast.Not)


class SingleAssignment:
    _count: int
    _rules: Rule

    def __init__(self) -> None:
        self._count = 0
        self._rules = many(
            _some_top_down(
                first(
                    [self._handle_return(), self._handle_for(), self._handle_assign()]
                )
            )
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

    def _fix_expr(
        self, prefix: str, value: Callable[[ast.AST], ast.AST]
    ) -> Callable[[ast.AST], ast.AST]:
        def _do_it(r: ast.AST) -> ast.AST:
            id = self._unique_id(prefix)
            return ast.Assign(
                targets=[ast.Name(id=id, ctx=ast.Store())], value=value(r)
            )

        return _do_it

    def _splice_non_identifier(
        self, original: List[ast.expr]
    ) -> Tuple[ast.Assign, List[ast.expr]]:
        id = self._unique_id("a")
        index, value = next(
            (i, v) for i, v in enumerate(original) if match(_not_identifier, v)
        )
        rewritten = (
            original[:index] + [ast.Name(id=id, ctx=ast.Load())] + original[index + 1 :]
        )
        assignment = ast.Assign(targets=[ast.Name(id=id, ctx=ast.Store())], value=value)
        return assignment, rewritten

    def _splice_non_identifier_keyword(
        self, original: List[ast.keyword]
    ) -> Tuple[ast.Assign, List[ast.keyword]]:
        id = self._unique_id("a")
        index, keyword = next(
            (i, k) for i, k in enumerate(original) if match(_not_identifier_keyword, k)
        )
        rewritten = (
            original[:index]
            + [ast.keyword(arg=keyword.arg, value=ast.Name(id=id, ctx=ast.Load()))]
            + original[index + 1 :]
        )
        assignment = ast.Assign(
            targets=[ast.Name(id=id, ctx=ast.Store())], value=keyword.value
        )
        return assignment, rewritten

    def _fix_call(self) -> Callable[[ast.Assign], ListEdit]:
        def _do_it(a: ast.Assign) -> ListEdit:
            c = a.value
            assert isinstance(c, ast.Call)
            assignment, args_new = self._splice_non_identifier(c.args)
            return ListEdit(
                [
                    assignment,
                    ast.Assign(
                        targets=a.targets,
                        value=ast.Call(func=c.func, args=args_new, keywords=c.keywords),
                    ),
                ]
            )

        return _do_it

    def _fix_call_keyword(self) -> Callable[[ast.Assign], ListEdit]:
        def _do_it(a: ast.Assign) -> ListEdit:
            c = a.value
            assert isinstance(c, ast.Call)
            assignment, keywords_new = self._splice_non_identifier_keyword(c.keywords)
            return ListEdit(
                [
                    assignment,
                    ast.Assign(
                        targets=a.targets,
                        value=ast.Call(func=c.func, args=c.args, keywords=keywords_new),
                    ),
                ]
            )

        return _do_it

    def _fix_list(self) -> Callable[[ast.Assign], ListEdit]:
        def _do_it(a: ast.Assign) -> ListEdit:
            c = a.value
            assert isinstance(c, ast.List)
            assignment, elts_new = self._splice_non_identifier(c.elts)
            return ListEdit(
                [
                    assignment,
                    ast.Assign(
                        targets=a.targets, value=ast.List(elts=elts_new, ctx=c.ctx)
                    ),
                ]
            )

        return _do_it

    def _handle_unassigned(self) -> Rule:  # unExp = unassigned expressions
        return PatternRule(
            expr(), self._fix_expr("u", lambda u: u.value), "handle_unassigned"
        )

    def _handle_return(self) -> Rule:
        return PatternRule(
            ast_return(value=_not_identifier),
            self._fix_it("r", lambda r: r.value, lambda r, v: ast.Return(value=v)),
            "handle_return",
        )

    def _handle_for(self) -> Rule:
        return PatternRule(
            ast_for(iter=_not_identifier),
            self._fix_it(
                "f",
                lambda f: f.iter,
                lambda f, v: ast.For(
                    target=f.target, iter=v, body=f.body, orelse=f.orelse
                ),
            ),
            "handle_for",
        )

    def _handle_assign(self) -> Rule:
        return pattern_rules(
            [
                (
                    assign(value=unaryop(operand=_not_identifier, op=_unops)),
                    self._fix_it(
                        "a",
                        lambda a: a.value.operand,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.UnaryOp(operand=v, op=a.value.op),
                        ),
                    ),
                ),
                # a = (b + c)[d + e] becomes t = b + c, a = t[d + e]
                (
                    assign(value=subscript(value=_not_identifier)),
                    self._fix_it(
                        "a",
                        lambda a: a.value.value,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.Subscript(
                                value=v, slice=a.value.slice, ctx=a.value.ctx
                            ),
                        ),
                    ),
                ),
                # TODO: Handle slices other than Index
                # a = b[d + e] becomes t = d + e, a = b[t]
                (
                    assign(value=subscript(slice=index(value=_not_identifier))),
                    self._fix_it(
                        "a",
                        lambda a: a.value.slice.value,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.Subscript(
                                value=a.value.value,
                                slice=ast.Index(value=v),
                                ctx=a.value.ctx,
                            ),
                        ),
                    ),
                ),
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
                # If we have t = foo.bar(...) rewrite that as t1 = foo.bar, t = t1(...)
                (
                    assign(value=call(func=_not_identifier)),
                    self._fix_it(
                        "a",
                        lambda a: a.value.func,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.Call(
                                func=v, args=a.value.args, keywords=a.value.keywords
                            ),
                        ),
                    ),
                ),
                # If we have t = foo(x + y, 2) rewrite that to
                # t1 = x + y, t2 = 2, t = foo(t1, t2).
                (
                    assign(value=call(func=name(), args=_list_not_identifier)),
                    self._fix_call(),
                ),
                # If we have t = foo(a, b, z=123) rewrite that to
                # t1 = 123, t = foo(a, b, t1),
                # but do it after we've rewriten the receiver and the
                # positional arguments.
                (
                    assign(
                        value=call(
                            func=name(),
                            args=_list_all_identifiers,
                            keywords=_not_identifier_keywords,
                        )
                    ),
                    self._fix_call_keyword(),
                ),
                # If we have t = (x + y).z, rewrite that as t1 = x + y, t = t1.z
                (
                    assign(value=attribute(value=_not_identifier)),
                    self._fix_it(
                        "a",
                        lambda a: a.value.value,
                        lambda a, v: ast.Assign(
                            targets=a.targets,
                            value=ast.Attribute(
                                value=v, attr=a.value.attr, ctx=a.value.ctx
                            ),
                        ),
                    ),
                ),
                (assign(value=ast_list(elts=_list_not_identifier)), self._fix_list()),
            ],
            "handle_assign",
        )

    def single_assignment(self, node: ast.AST) -> ast.AST:
        return self._rules(node).expect_success()


def single_assignment(node: ast.AST) -> ast.AST:
    s = SingleAssignment()
    return s.single_assignment(node)
