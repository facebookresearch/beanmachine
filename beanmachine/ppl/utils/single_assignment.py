#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
from typing import Callable, List, Tuple

from beanmachine.ppl.utils.ast_patterns import (
    assign,
    ast_boolop,
    ast_compare,
    ast_dict,
    ast_domain,
    ast_for,
    ast_if,
    ast_list,
    ast_return,
    ast_true,
    ast_while,
    attribute,
    binop,
    call,
    expr,
    index,
    keyword,
    match,
    match_any,
    name,
    starred,
    subscript,
    unaryop,
)
from beanmachine.ppl.utils.patterns import (
    HeadTail,
    ListAll,
    ListAny,
    Pattern,
    PatternBase,
    anyPattern,
    negate,
    twoPlusList,
)
from beanmachine.ppl.utils.rules import (
    FirstMatch as first,
    ListEdit,
    PatternRule,
    Rule,
    TryMany as many,
)


_some_top_down = ast_domain.some_top_down
_not_identifier: Pattern = negate(name())
_not_starred: Pattern = negate(starred())
_list_not_identifier: PatternBase = ListAny(_not_identifier)
_list_not_starred: PatternBase = ListAny(_not_starred)
_list_all_identifiers: PatternBase = ListAll(name())
_not_identifier_keyword: Pattern = keyword(value=_not_identifier)
_not_identifier_keywords: PatternBase = ListAny(_not_identifier_keyword)

# TODO: The identifier "dict" should be made global unique in target name space
_keyword_with_dict = keyword(arg=None, value=call(func=name(id="dict"), args=[]))
_keyword_with_no_arg = keyword(arg=None)
_not_keyword_with_no_arg = negate(_keyword_with_no_arg)
_list_not_keyword_with_no_arg = ListAny(_not_keyword_with_no_arg)
_keyword_with_arg = keyword(arg=negate(None))
_list_with_keyword_with_arg = ListAny(_keyword_with_arg)

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
                    [
                        self._handle_compare_all(),
                        self._handle_boolop_all(),
                        self._handle_while(),
                        self._handle_if(),
                        self._handle_unassigned(),
                        self._handle_return(),
                        self._handle_for(),
                        self._handle_assign(),
                    ]
                )
            )
        )

    def _unique_id(self, prefix: str) -> str:
        self._count = self._count + 1
        return f"{prefix}{self._count}"

    def _transform_with_name(
        self,
        prefix: str,
        extract_expr: Callable[[ast.AST], ast.expr],
        build_new_term: Callable[[ast.AST, ast.AST], ast.AST],
    ) -> Callable[[ast.AST], ListEdit]:
        def _do_it(r: ast.AST) -> ListEdit:
            id = self._unique_id(prefix)
            return ListEdit(
                [
                    ast.Assign(
                        targets=[ast.Name(id=id, ctx=ast.Store())],
                        value=extract_expr(r),
                    ),
                    build_new_term(r, ast.Name(id=id, ctx=ast.Load())),
                ]
            )

        return _do_it

    def _transform_with_assign(
        self,
        prefix: str,
        extract_expr: Callable[[ast.AST], ast.expr],
        build_new_term: Callable[[ast.AST, ast.AST, ast.AST], ListEdit],
    ) -> Callable[[ast.AST], ListEdit]:
        def _do_it(r: ast.AST) -> ListEdit:
            id = self._unique_id(prefix)
            new_assign = ast.Assign(
                targets=[ast.Name(id=id, ctx=ast.Store())], value=extract_expr(r)
            )
            return build_new_term(r, ast.Name(id=id, ctx=ast.Load()), new_assign)

        return _do_it

    def _transform_expr(
        self, prefix: str, extract_expr: Callable[[ast.AST], ast.expr]
    ) -> Callable[[ast.AST], ast.AST]:
        def _do_it(r: ast.AST) -> ast.AST:
            id = self._unique_id(prefix)
            return ast.Assign(
                targets=[ast.Name(id=id, ctx=ast.Store())], value=extract_expr(r)
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

    def _splice_non_entry(
        self, keys: List[ast.expr], values: List[ast.expr]
    ) -> Tuple[ast.Assign, List[ast.expr], List[ast.expr]]:
        id = self._unique_id("a")
        keyword_index, keyword = next(
            ((i, k) for i, k in enumerate(keys) if match(_not_identifier, k)),
            (len(keys), None),
        )
        value_index, value = next(
            ((i, v) for i, v in enumerate(values) if match(_not_identifier, v)),
            (len(values), None),
        )

        # pyre-fixme[16]: `None` has no attribute `__gt__`.
        # pyre-fixme[16]: `None` has no attribute `__le__`.
        if keyword_index <= value_index:
            keys_new = (
                keys[:keyword_index]
                + [ast.Name(id=id, ctx=ast.Load())]
                # pyre-fixme[6]: Expected `int` for 1st param but got `Union[None,
                #  _ast.expr, int]`.
                + keys[keyword_index + 1 :]
            )
            assignment = ast.Assign(
                targets=[ast.Name(id=id, ctx=ast.Store())], value=keyword
            )
            return assignment, keys_new, values
        else:
            values_new = (
                values[:value_index]
                + [ast.Name(id=id, ctx=ast.Load())]
                # pyre-fixme[6]: Expected `int` for 1st param but got `Union[None,
                #  _ast.expr, int]`.
                + values[value_index + 1 :]
            )
            assignment = ast.Assign(
                targets=[ast.Name(id=id, ctx=ast.Store())], value=value
            )
            return assignment, keys, values_new

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

    def _splice_non_starred(self, original: List[ast.expr]) -> List[ast.expr]:
        index, value = next(
            (i, v) for i, v in enumerate(original) if match(_not_starred, v)
        )
        rewritten = (
            original[:index]
            + [ast.Starred(ast.List(elts=[value], ctx=ast.Load()), ast.Load())]
            + original[index + 1 :]
        )

        return rewritten

    # TODO: The identifier "dict" should be made global unique in target name space
    def _splice_non_double_starred(
        self, original: List[ast.keyword]
    ) -> List[ast.keyword]:
        index, value = next(
            (i, v) for i, v in enumerate(original) if match(_keyword_with_arg, v)
        )
        rewritten = (
            original[:index]
            + [
                ast.keyword(
                    arg=None,
                    value=ast.Call(
                        func=ast.Name(id="dict", ctx=ast.Load()),
                        args=[],
                        keywords=[value],
                    ),
                )
            ]
            + original[index + 1 :]
        )

        return rewritten

    def _transform_call(self) -> Callable[[ast.Assign], ListEdit]:
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

    def _transform_call_keyword(self) -> Callable[[ast.Assign], ListEdit]:
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

    def _transform_list(  # TODO: Generalization of ast_op to Callable is tentative
        self, ast_op: Callable[[ast.Assign], type] = lambda a: ast.List
    ) -> Callable[[ast.Assign], ListEdit]:
        def _do_it(a: ast.Assign) -> ListEdit:
            c = a.value
            ast_op_a = ast_op(a)
            assert isinstance(c, ast_op_a)
            assignment, elts_new = self._splice_non_identifier(c.elts)
            return ListEdit(
                [
                    assignment,
                    ast.Assign(targets=a.targets, value=ast_op_a(elts_new, c.ctx)),
                ]
            )

        return _do_it

    def _transform_lists(  # For things like ast_op = ast.Dict
        self, ast_op: type = ast.Dict
    ) -> Callable[[ast.Assign], ListEdit]:
        def _do_it(a: ast.Assign) -> ListEdit:
            c = a.value
            assert isinstance(c, ast_op)
            assignment, keys_new, values_new = self._splice_non_entry(c.keys, c.values)
            return ListEdit(
                [
                    assignment,
                    ast.Assign(targets=a.targets, value=ast_op(keys_new, values_new)),
                ]
            )

        return _do_it

    def _handle_while_True(self) -> Rule:
        return PatternRule(
            ast_while(test=ast_true, orelse=negate([])),
            lambda source_term: ListEdit(
                [ast.While(test=source_term.test, body=source_term.body, orelse=[])]
            ),
            "handle_while_True",
        )

    def _handle_while_not_True(self) -> Rule:
        return PatternRule(
            ast_while(test=negate(ast_true)),
            self._transform_with_assign(
                "w",
                lambda source_term: source_term.test,
                lambda source_term, new_name, new_assign: ListEdit(
                    [
                        ast.While(
                            # TODO test=ast.Name(id="True", ctx=ast.Load()),
                            test=ast.NameConstant(value=True),
                            body=[
                                new_assign,
                                ast.If(test=new_name, body=source_term.body, orelse=[]),
                            ],
                            orelse=[],
                        ),
                        ast.If(
                            test=ast.UnaryOp(op=ast.Not(), operand=new_name),
                            body=source_term.orelse,
                            orelse=[],
                        ),
                    ]
                ),
            ),
            "handle_while_not_True",
        )

    def _handle_while(self) -> Rule:
        return first([self._handle_while_True(), self._handle_while_not_True()])

    def _handle_unassigned(self) -> Rule:  # unExp = unassigned expressions
        return PatternRule(
            expr(), self._transform_expr("u", lambda u: u.value), "handle_unassigned"
        )

    def _handle_return(self) -> Rule:
        return PatternRule(
            ast_return(value=_not_identifier),
            self._transform_with_name(
                "r",
                lambda source_term: source_term.value,
                lambda _, new_name: ast.Return(value=new_name),
            ),
            "handle_return",
        )

    def _handle_if(self) -> Rule:
        return PatternRule(
            ast_if(test=_not_identifier),
            self._transform_with_name(
                "r",
                lambda source_term: source_term.test,
                lambda source_term, new_name: ast.If(
                    test=new_name, body=source_term.body, orelse=source_term.orelse
                ),
            ),
            "handle_if",
        )

    def _handle_for(self) -> Rule:
        return PatternRule(
            ast_for(iter=_not_identifier),
            self._transform_with_name(
                "f",
                lambda source_term: source_term.iter,
                lambda source_term, new_name: ast.For(
                    target=source_term.target,
                    iter=new_name,
                    body=source_term.body,
                    orelse=source_term.orelse,
                ),
            ),
            "handle_for",
        )

    # Start of a series of rules that will define handle_assign
    def _handlle_assign_unaryop(self) -> Rule:
        return PatternRule(
            assign(value=unaryop(operand=_not_identifier, op=_unops)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.operand,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.UnaryOp(operand=new_name, op=source_term.value.op),
                ),
            ),
            "handle_assign_unaryop",
        )

    def _handle_assign_subscript(self) -> Rule:
        # a = (b + c)[d + e] becomes t = b + c, a = t[d + e]
        return PatternRule(
            assign(value=subscript(value=_not_identifier)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Subscript(
                        value=new_name,
                        slice=source_term.value.slice,
                        ctx=source_term.value.ctx,
                    ),
                ),
            ),
            "handle_assign_subscript",
        )

    def _handle_assign_subscript_slice(self) -> Rule:
        # TODO: Handle slices other than Index
        # a = b[d + e] becomes t = d + e, a = b[t]
        return PatternRule(
            assign(value=subscript(slice=index(value=_not_identifier))),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.slice.value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Subscript(
                        value=source_term.value.value,
                        slice=ast.Index(value=new_name),
                        ctx=source_term.value.ctx,
                    ),
                ),
            ),
            "handle_assign_subscript_slice",
        )

    def _handle_assign_binop_left(self) -> Rule:
        return PatternRule(
            assign(value=binop(left=_not_identifier, op=_binops)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.left,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.BinOp(
                        left=new_name,
                        op=source_term.value.op,
                        right=source_term.value.right,
                    ),
                ),
            ),
            "handle_assign_binop_left",
        )

    def _handle_assign_binop_right(self) -> Rule:
        return PatternRule(
            assign(value=binop(right=_not_identifier, op=_binops)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.right,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.BinOp(
                        left=source_term.value.left,
                        op=source_term.value.op,
                        right=new_name,
                    ),
                ),
            ),
            "handle_assign_binop_right",
        )

    def _handle_assign_call_function_expression(self) -> Rule:
        # If we have t = foo.bar(...) rewrite that as t1 = foo.bar, t = t1(...)
        return PatternRule(
            assign(value=call(func=_not_identifier)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.func,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Call(
                        func=new_name,
                        args=source_term.value.args,
                        keywords=source_term.value.keywords,
                    ),
                ),
            ),
            "handle_assign_call_function_expression",
        )

    def _handle_assign_call_single_star_arg(self) -> Rule:
        # Rewrite x = f(*([1]+[2]) into d=[1]+[2]; x = f(*d)
        return PatternRule(
            assign(value=call(func=name(), args=[starred(value=_not_identifier)])),
            self._transform_with_name(
                "r",
                lambda source_term: source_term.value.args[0].value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Call(
                        func=source_term.value.func,
                        args=[ast.Starred(new_name, source_term.value.args[0].ctx)],
                        keywords=source_term.value.keywords,
                    ),
                ),
            ),
            "handle_assign_call_single_star_arg",
        )

    def _handle_assign_call_single_double_star_arg(self) -> Rule:
        # Rewrite x = f(*a, **{k:5}) into t = {k: 5} ; x = f(*a, **t)
        # Note: In the strategy we have chosen for dealing with keywords
        #       the argument to ** should normally be dict(...). However,
        #       if there is only a single argument of the form above there
        #       is no need for the further checking and the short-cut
        #       transformation is therefore expected to be sound.
        return PatternRule(
            assign(
                value=call(
                    func=name(),
                    args=[starred(value=name())],
                    keywords=[keyword(arg=None, value=_not_identifier)],
                )
            ),
            self._transform_with_name(
                "r",
                lambda source_term: source_term.value.keywords[0].value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Call(
                        func=source_term.value.func,
                        args=source_term.value.args,
                        keywords=[ast.keyword(arg=None, value=new_name)],
                    ),
                ),
            ),
            "handle_assign_call_single_double_star_arg",
        )

    def _handle_assign_call_two_star_args(self) -> Rule:
        # Rewrite x= f(*[1],*[2]) into x = f(*([1]+[2]))
        # TODO: Ideally, would like to merge [1].ctx with the [0].ctx below
        return PatternRule(
            assign(
                value=call(args=HeadTail(starred(), HeadTail(starred(), anyPattern)))
            ),
            lambda source_term: ast.Assign(
                targets=source_term.targets,
                value=ast.Call(
                    func=source_term.value.func,
                    args=[
                        ast.Starred(
                            ast.BinOp(
                                left=source_term.value.args[0].value,
                                op=ast.Add(),
                                right=source_term.value.args[1].value,
                            ),
                            source_term.value.args[0].ctx,
                        )
                    ]
                    + source_term.value.args[2:],
                    keywords=source_term.value.keywords,
                ),
            ),
            "handle_assign_call_two_star_args",
        )

    def _handle_assign_call_two_double_star_args(self) -> Rule:
        # Rewrite x = f(**{a:1},**{b:3})
        #  into   x = f(**dict(**{a: 1}, **{b: 3}))
        # Note: Since we are not lifting, no restriction needed on func or args
        # TODO: Ideally, would like to merge [1].ctx with the [0].ctx below
        # TODO: The identifier "dict" should be made global unique in target name space
        return PatternRule(
            assign(
                value=call(
                    func=negate(name(id="dict")),
                    keywords=HeadTail(
                        _keyword_with_no_arg, HeadTail(_keyword_with_no_arg, anyPattern)
                    ),
                )
            ),
            lambda source_term: ast.Assign(
                targets=source_term.targets,
                value=ast.Call(
                    func=source_term.value.func,
                    args=source_term.value.args,
                    keywords=[
                        ast.keyword(
                            arg=None,
                            value=ast.Call(
                                func=ast.Name(id="dict", ctx=ast.Load()),
                                args=[],
                                keywords=[source_term.value.keywords[0]]
                                + [source_term.value.keywords[1]],
                            ),
                        )
                    ]
                    + source_term.value.keywords[2:],
                ),
            ),
            "handle_assign_call_two_double_star_args",
        )

    def _handle_assign_call_regular_arg(self) -> Rule:
        # Rewrite x = f(*[1],2) into x = f(*[1],*[2])
        return PatternRule(
            assign(value=call(args=_list_not_starred)),
            lambda source_term: ast.Assign(
                targets=source_term.targets,
                value=ast.Call(
                    func=source_term.value.func,
                    args=self._splice_non_starred(source_term.value.args),
                    keywords=source_term.value.keywords,
                ),
            ),
            "_handle_assign_call_regular_arg",
        )

    def _handle_assign_call_keyword_arg(self) -> Rule:
        # Rewrite x = f(k=42) into x = f(**dict(k=42))
        # TODO: The identifier "dict" should be made global unique in target name space

        return PatternRule(
            assign(
                value=call(
                    func=negate(name(id="dict")), keywords=_list_with_keyword_with_arg
                )
            ),
            lambda source_term: ast.Assign(
                targets=source_term.targets,
                value=ast.Call(
                    func=source_term.value.func,
                    args=source_term.value.args,
                    keywords=self._splice_non_double_starred(
                        source_term.value.keywords
                    ),
                ),
            ),
            "_handle_assign_call_keyword_arg",
        )

    def _handle_assign_call_empty_regular_arg(self) -> Rule:
        # Rewrite x = f(*[1],2) into x = f(*[1],*[2])
        # TODO: The identifier "dict" should be made global unique in target name space
        return PatternRule(
            assign(value=call(func=negate(name(id="dict")), args=[])),
            lambda source_term: ast.Assign(
                targets=source_term.targets,
                value=ast.Call(
                    func=source_term.value.func,
                    args=[ast.Starred(ast.List([], ast.Load()), ast.Load())],
                    keywords=source_term.value.keywords,
                ),
            ),
            "_handle_assign_call_empty_regular_arg",
        )

    def _handle_assign_call_empty_keyward_arg(self) -> Rule:
        # Rewrite x = f(1) into x = f(1,**{})
        # Basically, ensure that any call has at least one ** argument
        # TODO: The identifier "dict" should be made global unique in target name space
        return PatternRule(
            assign(value=call(func=negate(name(id="dict")), keywords=[])),
            lambda source_term: ast.Assign(
                targets=source_term.targets,
                value=ast.Call(
                    func=source_term.value.func,
                    args=source_term.value.args,
                    keywords=[
                        ast.keyword(arg=None, value=ast.Dict(keys=[], values=[]))
                    ],
                ),
            ),
            "_handle_assign_call_empty_keyword_arg",
        )

    def _handle_handle_assign_attribute(self) -> Rule:
        # If we have t = (x + y).z, rewrite that as t1 = x + y, t = t1.z
        return PatternRule(
            assign(value=attribute(value=_not_identifier)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Attribute(
                        value=new_name,
                        attr=source_term.value.attr,
                        ctx=source_term.value.ctx,
                    ),
                ),
            ),
            "handle_assign_attribute",
        )

    def _handle_assign_list(self) -> Rule:
        return PatternRule(
            assign(value=ast_list(elts=_list_not_identifier)),
            self._transform_list(),
            "handle_assign_list",
        )

    def _handle_assign_tuple(self) -> Rule:
        return PatternRule(
            assign(value=ast_list(elts=_list_not_identifier, ast_op=ast.Tuple)),
            self._transform_list(ast_op=lambda a: ast.Tuple),
            "handle_assign_tuple",
        )

    def _handle_assign_dictionary_keys(self) -> Rule:
        return PatternRule(
            assign(value=ast_dict(keys=_list_not_identifier)),
            self._transform_lists(),
            "handle_assign_dictionary_keys",
        )

    def _handle_assign_dictionary_values(self) -> Rule:
        return PatternRule(
            assign(value=ast_dict(values=_list_not_identifier)),
            self._transform_lists(),
            "handle_assign_dictionary_values",
        )

    def _handle_assign(self) -> Rule:
        return first(
            [
                self._handlle_assign_unaryop(),
                self._handle_assign_subscript(),
                self._handle_assign_subscript_slice(),
                self._handle_assign_binop_left(),
                self._handle_assign_binop_right(),
                self._handle_handle_assign_attribute(),
                self._handle_assign_list(),
                self._handle_assign_tuple(),
                self._handle_assign_dictionary_keys(),
                self._handle_assign_dictionary_values(),
                # Acceptable rules for handling function calls
                self._handle_assign_call_function_expression(),
                #  Rules for regular arguments
                self._handle_assign_call_single_star_arg(),
                self._handle_assign_call_two_star_args(),
                self._handle_assign_call_regular_arg(),
                self._handle_assign_call_empty_regular_arg(),
                #  Rules for keyword arguments
                self._handle_assign_call_single_double_star_arg(),
                self._handle_assign_call_two_double_star_args(),
                self._handle_assign_call_keyword_arg(),
                self._handle_assign_call_empty_keyward_arg(),
            ]
        )

    def _handle_boolop_binarize(self) -> Rule:
        return PatternRule(
            ast_boolop(values=twoPlusList),
            lambda source_term: ast.BoolOp(
                op=source_term.op,
                values=[
                    ast.BoolOp(
                        source_term.op, [source_term.values[0], source_term.values[1]]
                    )
                ]
                + source_term.values[2:],
            ),
            "handle_boolop_binarize",
        )

    def _handle_assign_boolop_linearize(self) -> Rule:
        return PatternRule(  # a = e1 and e2 rewrites into b = e1, a = b and e2
            assign(value=ast_boolop(values=[_not_identifier, anyPattern])),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.values[0],
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.BoolOp(
                        op=source_term.value.op,
                        values=[new_name, source_term.value.values[1]],
                    ),
                ),
            ),
            "handle_assign_boolop_linearize",
        )

    def _handle_assign_and2if(self) -> Rule:
        return PatternRule(
            assign(value=ast_boolop(op=ast.And, values=[name(), anyPattern])),
            lambda source_term: ast.If(
                test=source_term.value.values[0],
                body=[
                    ast.Assign(
                        targets=source_term.targets, value=source_term.value.values[1]
                    )
                ],
                orelse=[
                    ast.Assign(
                        targets=source_term.targets, value=source_term.value.values[0]
                    )
                ],
            ),
            "handle_and2if",
        )

    def _handle_assign_or2if(self) -> Rule:
        return PatternRule(
            assign(value=ast_boolop(op=ast.Or, values=[name(), anyPattern])),
            lambda source_term: ast.If(
                test=source_term.value.values[0],
                body=[
                    ast.Assign(
                        targets=source_term.targets, value=source_term.value.values[0]
                    )
                ],
                orelse=[
                    ast.Assign(
                        targets=source_term.targets, value=source_term.value.values[1]
                    )
                ],
            ),
            "handle_or2if",
        )

    def _handle_boolop_all(self) -> Rule:
        return first(
            [
                self._handle_boolop_binarize(),
                self._handle_assign_boolop_linearize(),
                self._handle_assign_and2if(),
                self._handle_assign_or2if(),
            ]
        )

    def _handle_compare_binarize(self) -> Rule:
        # Rewrite things like x = a < b > c ... to x = a < b and b > c ...
        return PatternRule(
            ast_compare(
                ops=HeadTail(anyPattern, HeadTail(anyPattern, anyPattern)),
                comparators=HeadTail(name(), anyPattern),
            ),
            lambda source_term: ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=source_term.left,
                        ops=[source_term.ops[0]],
                        comparators=[source_term.comparators[0]],
                    ),
                    ast.Compare(
                        left=source_term.comparators[0],
                        ops=source_term.ops[1:],
                        comparators=source_term.comparators[1:],
                    ),
                ],
            ),
            "handle_compare_binarize",
        )

    def _handle_assign_compare_lefthandside(self) -> Rule:
        # Rewrite things like x = 1 + a < b ... to y = 1 + a; x = y < b ...
        return PatternRule(
            assign(value=ast_compare(left=_not_identifier)),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.left,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Compare(
                        left=new_name,
                        ops=source_term.value.ops,
                        comparators=source_term.value.comparators,
                    ),
                ),
            ),
            "handle_assign_compare_lefthandside",
        )

    def _handle_assign_compare_righthandside(self) -> Rule:
        # Rewrite things like x = a < 1 + b ... to y = 1 + b; x = a < y ...
        return PatternRule(
            assign(
                value=ast_compare(
                    left=name(), comparators=HeadTail(_not_identifier, anyPattern)
                )
            ),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.comparators[0],
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Compare(
                        left=source_term.value.left,
                        ops=source_term.value.ops,
                        comparators=[new_name] + source_term.value.comparators[1:],
                    ),
                ),
            ),
            "handle_assign_compare_righthandside",
        )

    def _handle_compare_all(self) -> Rule:
        return first(
            [
                self._handle_compare_binarize(),
                self._handle_assign_compare_righthandside(),
                self._handle_assign_compare_lefthandside(),
            ]
        )

    def single_assignment(self, node: ast.AST) -> ast.AST:
        return self._rules(node).expect_success()


def single_assignment(node: ast.AST) -> ast.AST:
    s = SingleAssignment()
    return s.single_assignment(node)
