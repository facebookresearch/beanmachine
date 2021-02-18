#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

# This code transforms Python programs into a much simpler subset of Python with
# the same semantics. Some invariants of the simpler language:
#
# * All "return" statements return either nothing or an identifier.
# * All "while" loops are of the form "while True:"
# * No "while" loop has an "else" clause.
# * All "for" statements have an identifier as their collection.
# * All "if" statements have an identifier as their condition.
# * There are no statements that are just a single expression.
#
# * All unary operators (+, -, ~, not) have an identifier as their operand.
# * All binary operators (+, -, *, /, //, %, **, <<, >>, |, ^, &, @) have an identifier
#   as both operands.
# * There are no "and" or "or" operators
# * All comparison operators (<, >, <=, >=, ==, !=, is, is not, in, not in)
#   are binary operators where both operands are identifiers.
# * All indexing operators (a[b]) have identifiers as both collection and index.
# * The left side of all attribute accesses ("dot") is an identifier.
#   That is "id.attr".
# * Every literal list contains only identifiers. That is "[id, id, id, ...]"
# * Every literal dictionary consists only of identifiers for both keys and values.
#   That is "{id : id, id : id, ... }"
# * Every function call is of the exact form "id(*id, **id)". There are no "normal"
#   arguments. There are some exceptions to this rule:
#   * dict() is allowed. (TODO: this could be rewritten to {})
#   * dict(key = id) is allowed.
#   * dict(**id, **id) is allowed.
#   * TODO: There are similar exceptions for set and list; say what they are.
# * There are no dictionary, list or set comprehensions; they are rewritten as loops.
# TODO: say something about assert, delete, pass, import, break, continue, try, with.
# TODO: say something about global / nonlocal
# TODO: say something about yield
# TODO: say something about classes
# TODO: say something about nested functions
# TODO: say something about decorators
# TODO: say something about type annotations
# TODO: say something about lambdas
# TODO: say something about async
# TODO: say something about conditional expressions
# TODO: say something about formatted strings


import ast
from typing import Any, Callable, List, Tuple

from beanmachine.ppl.utils.ast_patterns import (
    assign,
    ast_boolop,
    ast_compare,
    ast_dict,
    ast_dictComp,
    ast_domain,
    ast_for,
    ast_if,
    ast_list,
    ast_listComp,
    ast_luple,
    ast_return,
    ast_setComp,
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
    match_every,
    name,
    slice_pattern,
    starred,
    subscript,
    unaryop,
)
from beanmachine.ppl.utils.beanstalk_common import allowed_functions
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
_name_or_none = match_any(name(), None)
_neither_name_nor_none: Pattern = negate(_name_or_none)
_not_starred: Pattern = negate(starred())
_list_not_identifier: PatternBase = ListAny(_not_identifier)
_list_not_starred: PatternBase = ListAny(_not_starred)
_list_all_identifiers: PatternBase = ListAll(name())
_not_identifier_keyword: Pattern = keyword(value=_not_identifier)
_not_identifier_keywords: PatternBase = ListAny(_not_identifier_keyword)
_not_none = negate(None)

# TODO: The identifier "dict" should be made global unique in target name space
_keyword_with_dict = keyword(arg=None, value=call(func=name(id="dict"), args=[]))
_keyword_with_no_arg = keyword(arg=None)
_not_keyword_with_no_arg = negate(_keyword_with_no_arg)
_list_not_keyword_with_no_arg = ListAny(_not_keyword_with_no_arg)
_keyword_with_arg = keyword(arg=negate(None))
_list_with_keyword_with_arg = ListAny(_keyword_with_arg)
# _not_in_allowed_functions: Pattern = negate(name(id="dict"))
_not_in_allowed_functions: Pattern = negate(
    match_any(*[name(id=t.__name__) for t in allowed_functions])
)

# TODO: Add MatMult to this list
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
    # TODO: Better naming convention. In particular, _rules is the reflexive
    # transitive congruent extension of _rule. It would be nice to find
    # terminology or refactoring that would make this easy to express more
    # clearly than "s" does, but while still being concise.
    _rule: Rule
    _rules: Rule

    def __init__(self) -> None:
        self._count = 0
        self._rule = first(
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
        self._rules = many(_some_top_down(self._rule))

    def _unique_id(self, prefix: str) -> str:
        self._count = self._count + 1
        return f"{prefix}{self._count}"

    def fresh_names(
        self,
        prefix_list: List[str],
        builder: Callable[[Callable[[str, str], ast.Name]], Any],
    ) -> Any:
        # This function gives us a way to treat a list of new local variables by their
        # original name, while giving them fresh names in the generated code to avoid name clashes
        # TODO: In the type this function, both instances of the type Any should
        # simply be the same type. It would be nice if there was a good way to use type variables
        # with Python
        id = {prefix: self._unique_id(prefix) for prefix in prefix_list}
        new_name_store = {
            (p, "store"): ast.Name(id=id[p], ctx=ast.Store()) for p in prefix_list
        }
        new_name_load = {
            (p, "load"): ast.Name(id=id[p], ctx=ast.Load()) for p in prefix_list
        }
        new_name = {**new_name_store, **new_name_load}
        return builder(lambda prefix, hand_side: new_name[(prefix, hand_side)])

    def _transform_with_name(
        self,
        prefix: str,
        extract_expr: Callable[[ast.AST], ast.expr],
        build_new_term: Callable[[ast.AST, ast.AST], ast.AST],
        extract_pattern: Callable[[ast.AST, ast.AST], List[ast.expr]] = lambda s, n: [
            n
        ],
    ) -> Callable[[ast.AST], ListEdit]:
        # Given its arguments (p,e,b) produces a term transformer
        #   r -> p_i = e(r) ; b(r,p_i) where p_i is a new name
        def _do_it(r: ast.AST) -> ListEdit:
            id = self._unique_id(prefix)
            return ListEdit(
                [
                    ast.Assign(
                        targets=extract_pattern(r, ast.Name(id=id, ctx=ast.Store())),
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
        # Given its arguments (p,e,b) produces a term transformer
        #   r -> b(r,p_i,p_i = e(r)) where p_i is a new name
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
        # Given its arguments (p,e) produces a term transformer
        #   r -> p_i = e(r) where p_i is a new name
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

        if keyword_index <= value_index:
            keys_new = (
                keys[:keyword_index]
                + [ast.Name(id=id, ctx=ast.Load())]
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

    def _handle_while_True_else(self) -> Rule:
        # This rule eliminates a redundant "else" clause from a "while True:" statement.
        # The purpose of this rule will become clear upon examining the rule which follows.
        #
        # Python has a seldom-used structure:
        # while condition:
        #   body
        # else:
        #   alternative
        #
        # The alternative is only executed if the condition is ever tested and False.
        # That is, if the loop is exited because of a break, return, or raised exception,
        # then the alternative is not executed.
        #
        # Obviously an else clause on a "while True" cannot be executed; though this is
        # rare, if we encounter it we can simply eliminate the "else:" entirely.
        return PatternRule(
            ast_while(test=ast_true, orelse=negate([])),
            lambda source_term: ListEdit(
                [ast.While(test=source_term.test, body=source_term.body, orelse=[])]
            ),
            "handle_while_True_else",
        )

    def _handle_while_not_True_else(self) -> Rule:
        # This rule eliminates all "while condition:" statements where the condition is
        # not "True", and there is an "else" clause. We rewrite
        #
        # while condition:
        #   body
        # else:
        #   alternative
        #
        # to
        #
        # while True:
        #   t = condition
        #   if t:
        #     body
        #   else:
        #     break
        # if not t:
        #   alternative
        #
        # which has the same semantics.
        #
        return PatternRule(
            ast_while(test=negate(ast_true), orelse=negate([])),
            self._transform_with_assign(
                "w",
                lambda source_term: source_term.test,
                lambda source_term, new_name, new_assign: ListEdit(
                    [
                        ast.While(
                            test=ast.NameConstant(value=True),
                            body=[
                                new_assign,
                                ast.If(
                                    test=new_name,
                                    body=source_term.body,
                                    orelse=[ast.Break()],
                                ),
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
            "handle_while_not_True_else",
        )

    def _handle_while_not_True(self) -> Rule:
        # This rule eliminates all "while condition:" statements where the condition is
        # not "True", and there is no "else" clause. We rewrite
        #
        # while condition:
        #   body
        #
        # to
        #
        # while True:
        #   t = condition
        #   if t:
        #     body
        #   else:
        #     break
        #
        # which has the same semantics.
        #
        return PatternRule(
            ast_while(test=negate(ast_true), orelse=[]),
            self._transform_with_assign(
                "w",
                lambda source_term: source_term.test,
                lambda source_term, new_name, new_assign: ListEdit(
                    [
                        ast.While(
                            test=ast.NameConstant(value=True),
                            body=[
                                new_assign,
                                ast.If(
                                    test=new_name,
                                    body=source_term.body,
                                    orelse=[ast.Break()],
                                ),
                            ],
                            orelse=[],
                        ),
                    ]
                ),
            ),
            "handle_while_not_True",
        )

    def _handle_while(self) -> Rule:
        # This rule eliminates all "else" clauses from while statements and
        # makes every while of the form "while True". See above for details.
        return first(
            [
                self._handle_while_True_else(),
                self._handle_while_not_True_else(),
                self._handle_while_not_True(),
            ]
        )

    def _handle_unassigned(self) -> Rule:
        # This rule eliminates all expressions that are used only for their side
        # effects and produces a redundant assignment. This is because a great
        # many other rules are defined in terms of rewriting assignments, and it
        # is easier to just turn unassigned values into assignments than to change
        # all those rules.  It rewrites:
        #
        # complex
        #
        # to
        #
        # t = complex
        return PatternRule(
            expr(), self._transform_expr("u", lambda u: u.value), "handle_unassigned"
        )

    def _handle_return(self) -> Rule:
        # This rule eliminates all "return" statements where there is a returned
        # value that is not an identifier. We rewrite:
        #
        # return complex
        #
        # to
        #
        # t = complex
        # return t
        #
        # TODO: Should we also eliminate plain returns? We could rewrite
        #
        # return
        #
        # as
        #
        # t = None
        # return t
        #
        # and thereby maintain the invariant that every return statement
        # returns an identifier.
        return PatternRule(
            ast_return(value=match_every(_not_identifier, _not_none)),
            self._transform_with_name(
                "r",
                lambda source_term: source_term.value,
                lambda _, new_name: ast.Return(value=new_name),
            ),
            "handle_return",
        )

    def _handle_if(self) -> Rule:
        # This rule eliminates all "if" statements where the condition is not
        # an identifier. It rewrites:
        #
        # if complex:
        #   consequence
        # else:
        #   alternative
        #
        # to
        #
        # t = complex
        # if t:
        #   consequence
        # else:
        #   alternative
        #
        # TODO: We can go further than this and eliminate all else clauses
        # from the simplified language by:
        #
        # t1 = bool(complex)
        # if t1:
        #   consequence
        # t2 = not t1
        # if t2:
        #   alternative
        #
        # Note that we've inserted a call to bool() above. The reason for that
        # is to ensure that we convert "complex" to bool *once* in this rewrite,
        # just as it is converted to bool *once* in the original code. The "not"
        # operator is defined as converting its operand to bool if it is not already
        # a bool, and then inverting the result.
        #
        # In addition to further simplifying the language, we will probably need
        # this proposed rewrite in order to make stochastic conditional control flows
        # work properly.

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
        # This eliminates all "for" statements where the collection is not an identifier.
        # It rewrites:
        #
        # for id in complex: ...
        #
        # to
        #
        # t = complex
        # for id in t: ...
        #
        # TODO: the "for" loop in Python supports an "else" clause which is only activated
        # when the loop is exited via "break". We could eliminate it.
        #
        # TODO: the "for" loop could be rewritten as fetching an iterator and iterating
        # over it until an exception is raised, but that's a rather complicated rewrite
        # and it might not be necessary to do so.

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

    #
    # Start of a series of rules that will define handle_assign
    #

    def _handle_assign_unaryop(self) -> Rule:
        # This rule eliminates all assignments where the right hand side
        # is a unary operator whose operand is not an identifier.
        # It rewrites:
        #
        # x = -complex
        #
        # to:
        #
        # t = complex
        # x = -t

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

    def _handle_assign_unary_dict(self) -> Rule:
        # This rule eliminates explicit call-style dictionary constructions where
        # there is a single argument and the value or collection is not an identifier.
        #
        # That is, We rewrite:
        #
        # x = dict(n = complex)
        #
        # to
        #
        # t = complex
        # x = dict(n = t)
        #
        # and from
        #
        # x = dict(**complex)
        #
        # to
        #
        # t = complex
        # x = dict(**complex)

        return PatternRule(
            assign(
                value=call(
                    func=name(id="dict"),
                    args=[],
                    keywords=[keyword(value=_not_identifier)],
                )
            ),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.keywords[0].value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Call(
                        func=source_term.value.func,  # Name(id="dict", ctx=Load()),
                        args=source_term.value.args,  # [],
                        keywords=[
                            ast.keyword(
                                arg=source_term.value.keywords[0].arg,  # "name",
                                value=new_name,
                            )
                        ],
                    ),
                ),
            ),
            "handle_assign_dict",
        )

    def _handle_assign_subscript_slice_all(self) -> Rule:
        # This rule address the simplification of subscript operations.
        return first(
            [
                self._handle_assign_subscript_slice_index_1(),
                self._handle_assign_subscript_slice_index_2(),
                self._handle_assign_subscript_slice_lower(),
            ]
        )

    def _handle_assign_subscript_slice_index_1(self) -> Rule:
        # This rule eliminates indexing expressions where the collection
        # indexed is not an identifier. We rewrite:
        #
        # x = complex[anything]
        #
        # to
        #
        # t = complex
        # x = t[anything]
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
            "handle_assign_subscript_slice_index_1",
        )

    def _handle_assign_subscript_slice_index_2(self) -> Rule:
        # This rule eliminates indexing expressions where the collection
        # indexed is an identifier but the index is not. We rewrite:
        #
        # x = c[complex]
        #
        # to
        #
        # t = complex
        # x = c[t]
        #
        # TODO: We do not yet handle slices other than straightforward
        # indices; we should also handle the other kinds of slices.
        return PatternRule(
            assign(value=subscript(value=name(), slice=index(value=_not_identifier))),
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
            "handle_assign_subscript_slice_index_2",
        )

    def _handle_assign_subscript_slice_lower(self) -> Rule:
        """Rewrites like e = a[b.c:] → x = b.c; e = a[x:]."""
        return PatternRule(
            assign(
                value=subscript(
                    value=name(), slice=slice_pattern(lower=_neither_name_nor_none)
                )
            ),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.slice.lower,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Subscript(
                        value=source_term.value.value,
                        slice=ast.Slice(
                            lower=ast.Name(id=new_name, ctx=ast.Load()),
                            upper=source_term.value.slice.upper,
                            step=source_term.value.slice.step,
                        ),
                        ctx=ast.Store(),
                    ),
                ),
            ),
            "_handle_assign_subscript_slice_lower",
        )

    def _handle_assign_binop_left(self) -> Rule:
        # This rule eliminates binary operators where the left hand side is not
        # an identifier. We rewrite:
        #
        # x = complex + anything
        #
        # to
        #
        # t = complex
        # x = t + anything
        #
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

    def _handle_assign_binary_dict_left(self) -> Rule:
        # This rule eliminates explicit call-style dictionary constructions where
        # there are exactly two arguments and the left value or collection is not
        # an identifier.
        #
        # We rewrite:
        #
        # x = dict(n1 = complex, anything)
        #
        # to
        #
        # t = complex
        # x = dict(n1 = t, anything)
        #
        # and we rewrite
        #
        # x = dict(**complex, anything)
        #
        # to
        #
        # t = complex
        # x = dict(**t, anything)

        return PatternRule(
            assign(
                value=call(
                    func=name(id="dict"),
                    args=[],
                    keywords=[keyword(value=_not_identifier), anyPattern],
                )
            ),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.keywords[0].value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Call(
                        func=source_term.value.func,  # Name(id="dict", ctx=Load()),
                        args=source_term.value.args,  # [],
                        keywords=[
                            ast.keyword(
                                arg=source_term.value.keywords[0].arg,  # "name",
                                value=new_name,
                            )
                        ]
                        + source_term.value.keywords[1:],
                    ),
                ),
            ),
            "handle_assign_binary_dict_left",
        )

    def _handle_assign_binary_dict_right(self) -> Rule:
        # This rule eliminates explicit call-style dictionary constructions where
        # there are exactly two arguments and the left value or collection is
        # an identifier but the right is not.
        #
        # Suppose "left" is "n1 = id" or "**id". We rewrite these:
        #
        # x = dict(left, n2 = complex)
        #
        # or
        #
        # x = dict(left, **complex)
        #
        # to
        #
        # t = complex
        # x = dict(left, n2 = t)
        #
        # or
        #
        # t = complex
        # x = dict(left, **t)
        return PatternRule(
            assign(
                value=call(
                    func=name(id="dict"),
                    args=[],
                    keywords=[keyword(value=name()), keyword(value=_not_identifier)],
                )
            ),
            self._transform_with_name(
                "a",
                lambda source_term: source_term.value.keywords[1].value,
                lambda source_term, new_name: ast.Assign(
                    targets=source_term.targets,
                    value=ast.Call(
                        func=source_term.value.func,  # Name(id="dict", ctx=Load()),
                        args=source_term.value.args,  # [],
                        keywords=source_term.value.keywords[:1]
                        + [
                            ast.keyword(
                                arg=source_term.value.keywords[1].arg,  # "name",
                                value=new_name,
                            )
                        ],
                    ),
                ),
            ),
            "handle_assign_binary_dict_right",
        )

    def _handle_assign_binop_right(self) -> Rule:
        # This rule eliminates binary operators where the left hand side is
        # an identifier but the right is not. We rewrite:
        #
        # x = id + anything
        #
        # to
        #
        # t = complex
        # x = id + t
        #
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
        # This rule eliminates function calls where the receiver is not an identifier.
        # We rewrite:
        #
        # x = complex(args)
        #
        # to
        #
        # t = complex
        # x = t(args)
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
        # This rule eliminates function calls of the form "id(*complex).
        # We rewrite:
        #
        # x = f(*complex)
        #
        # to
        #
        # t = complex
        # x = f(*t)
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
        # This rule eliminates function calls of the form "id(*id, **complex)".
        # We rewrite:
        #
        # x = f(*a, **complex)
        #
        # to
        #
        # t = complex
        # x = f(*a, **t)
        #
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
        # This rule eliminates function calls whose argument lists begin with
        # two starred arguments. We rewrite:
        #
        # x = f(*a1, *a2, ...)
        #
        # to
        #
        # x = f(*(a1 + a2), ...)
        #
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
        # This rule eliminates the leftmost pair of double-star arguments from function calls.
        # Here d1 and d2 are any expressions:
        #
        # x = f(a1, a2, ... , **d1, **d2, ...)
        #
        # to
        #
        # x = f(a1, a2, ..., **(dict(**d1, **d2)), ...)
        #
        # Note: Since we are not lifting, no restriction needed on func or args
        # TODO: Ideally, would like to merge [1].ctx with the [0].ctx below
        # TODO: The identifier "dict" should be made global unique in target name space
        return PatternRule(
            assign(
                value=call(
                    func=_not_in_allowed_functions,
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
        # This rule eliminates the leftmost non-starred argument from
        # a function argument list. We rewrite:
        #
        # x = f(*a1, *a2, anything, ...)
        #
        # to
        #
        # x = f(*a1, *a2, *[anything], ...)
        #
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
        # This rule eliminates a named argument from a function call by transforming
        # it into a double-starred argument; another rule then simplifies the double-
        # starred argument. We rewrite:
        #
        # x = f(... k = anything)
        #
        # to
        #
        # x = f(**dict(k = anything))
        #
        # TODO: The identifier "dict" should be made global unique in target name space

        return PatternRule(
            assign(
                value=call(
                    func=_not_in_allowed_functions, keywords=_list_with_keyword_with_arg
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
        # This rule eliminates function calls with empty non-named argument lists.
        # This guarantees that every function call has at least one unnamed argument.
        # We rewrite:
        #
        # x = f(only_named_or_double_starred_args)
        #
        # to:
        #
        # x = f(*[], only_named_or_double_starred_args)

        return PatternRule(
            assign(value=call(func=_not_in_allowed_functions, args=[])),
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

    def _handle_assign_call_empty_keyword_arg(self) -> Rule:
        # This rule eliminates function calls with no ** arguments. That is,
        # it ensures that every function call has at least one double-starred
        # argument. We rewrite:
        #
        # x = f(no_double_star_arguments)
        #
        # to
        #
        # x = f(no_double_star_arguments, **{})
        #
        # TODO: The identifier "dict" should be made global unique in target name space
        return PatternRule(
            assign(value=call(func=_not_in_allowed_functions, keywords=[])),
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

    def _handle_assign_attribute(self) -> Rule:
        # This rule eliminates attribute lookup ("dot") where the object
        # of the lookup is not an identifier. We rewrite:
        #
        # x = complex.z
        #
        # to:
        #
        # t = complex
        # x = t.z
        #
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
        # This rule eliminates the leftmost non-identifier from a list.
        # We rewrite:
        #
        # x = [ids, complex, ...]
        #
        # to:
        #
        # t = complex
        # x = [ids, t, ...]
        return PatternRule(
            assign(value=ast_list(elts=_list_not_identifier)),
            self._transform_list(),
            "handle_assign_list",
        )

    def _handle_assign_tuple(self) -> Rule:
        # This rule eliminates the leftmost non-identifier from a tuple.
        # We rewrite:
        #
        # x = (ids, complex, ...)
        #
        # to:
        #
        # t = complex
        # x = (ids, t, ...)
        return PatternRule(
            assign(value=ast_list(elts=_list_not_identifier, ast_op=ast.Tuple)),
            self._transform_list(ast_op=lambda a: ast.Tuple),
            "handle_assign_tuple",
        )

    def _handle_assign_dictionary_keys(self) -> Rule:
        # This rule eliminates the leftmost non-identifier dictionary key.
        # We rewrite:
        #
        # x = { complex : anything }
        #
        # to:
        #
        # t = complex
        # x = { t : anything }
        return PatternRule(
            assign(value=ast_dict(keys=_list_not_identifier)),
            self._transform_lists(),
            "handle_assign_dictionary_keys",
        )

    def _handle_assign_dictionary_values(self) -> Rule:
        # This rule eliminates the leftmost non-identifier dictionary value.
        # We rewrite:
        #
        # x = { anything : complex }
        #
        # to:
        #
        # t = complex
        # x = { anything : t }
        #
        # TODO: Note that this rule combined with the previous rule
        # changes the order in which side effects happen. If we have
        #
        # x = { a() : b(), c() : d() }
        #
        # Then this could be rewritten to:
        #
        # t1 = a()
        # t2 = c()
        # t3 = b()
        # t4 = d()
        # x = { t1 : t3, t2 : t4 }
        #
        # Which is not the a(), b(), c(), d() order we expect.
        #
        # We might consider fixing these rules so that the leftmost
        # key-or-value is rewritten, not the leftmost key and then the
        # leftmost value. However, this is low priority, as it is rare
        # for there to be a side effect in a dictionary key.

        return PatternRule(
            assign(value=ast_dict(values=_list_not_identifier)),
            self._transform_lists(),
            "handle_assign_dictionary_values",
        )

    def _nested_ifs_of(self, conditions: List[ast.expr], call: ast.stmt) -> ast.stmt:
        # Turns a series of conditions into nested ifs
        if conditions == []:
            return call
        else:
            head, *tail = conditions
            rest = self._nested_ifs_of(tail, call)
            return ast.If(test=head, body=[rest], orelse=[])

    def _nested_fors_and_ifs_of(
        self, generators: List[ast.comprehension], call: ast.stmt
    ) -> ast.stmt:
        # Turns nested comprehension generators into a nesting for for+if statements
        # for example [... for i in range(1,2) if odd(i)] into
        # for i in range(1,2):
        #    if odd(i):
        #       ...
        if generators == []:
            return call
        else:
            head, *tail = generators
            rest = self._nested_fors_and_ifs_of(tail, call)
            return ast.For(
                target=head.target,
                iter=head.iter,
                body=[self._nested_ifs_of(head.ifs, rest)],
                orelse=[],
                type_comment=None,
            )

    def _handle_assign_listComp(self) -> Rule:
        # Rewrite y = [c for v_i in e_i if b_i] into
        # def p():
        #    r = []
        #    for v_i in e_i
        #       if b_i:
        #          r.append(c)
        #    return r
        # y=p()
        _empty_ast_arguments = ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return PatternRule(
            assign(value=ast_listComp()),
            lambda term: ListEdit(
                self.fresh_names(
                    ["p", "r"],
                    lambda new_name: [
                        ast.FunctionDef(
                            name=new_name("p", "store").id,
                            args=_empty_ast_arguments,
                            body=[
                                ast.Assign(
                                    targets=[new_name("r", "store")],
                                    value=ast.List([], ast.Load()),
                                ),
                                self._nested_fors_and_ifs_of(
                                    term.value.generators,
                                    ast.Expr(
                                        ast.Call(
                                            func=ast.Attribute(
                                                value=new_name("r", "load"),
                                                attr="append",
                                                ctx=ast.Load(),
                                            ),
                                            args=[term.value.elt],
                                            keywords=[],
                                        )
                                    ),
                                ),
                                ast.Return(new_name("r", "load")),
                            ],
                            decorator_list=[],
                            returns=None,
                            type_comment=None,
                        ),
                        ast.Assign(
                            targets=term.targets,
                            value=ast.Call(
                                func=new_name("p", "load"), args=[], keywords=[]
                            ),
                        ),
                    ],
                )
            ),
            "handle_assign_listComp",
        )

    def _handle_assign_setComp(self) -> Rule:
        # Rewrite y = {c for v_i in e_i if b_i} into
        # def p():
        #    r = set()
        #    for v_i in e_i
        #       if b_i:
        #          r.add(c)
        #    return r
        # y=p()
        _empty_ast_arguments = ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return PatternRule(
            assign(value=ast_setComp()),
            lambda term: ListEdit(
                self.fresh_names(
                    ["p", "r"],
                    lambda new_name: [
                        ast.FunctionDef(
                            name=new_name("p", "store").id,
                            args=_empty_ast_arguments,
                            body=[
                                ast.Assign(
                                    targets=[new_name("r", "store")],
                                    value=ast.Call(
                                        func=ast.Name(id="set", ctx=ast.Load()),
                                        args=[],
                                        keywords=[],
                                    ),
                                ),
                                self._nested_fors_and_ifs_of(
                                    term.value.generators,
                                    ast.Expr(
                                        ast.Call(
                                            func=ast.Attribute(
                                                value=new_name("r", "load"),
                                                attr="add",
                                                ctx=ast.Load(),
                                            ),
                                            args=[term.value.elt],
                                            keywords=[],
                                        )
                                    ),
                                ),
                                ast.Return(new_name("r", "load")),
                            ],
                            decorator_list=[],
                            returns=None,
                            type_comment=None,
                        ),
                        ast.Assign(
                            targets=term.targets,
                            value=ast.Call(
                                func=new_name("p", "load"), args=[], keywords=[]
                            ),
                        ),
                    ],
                )
            ),
            "handle_assign_setComp",
        )

    def _handle_assign_dictComp(self) -> Rule:
        # Rewrite y = {c:d for v_i in e_i if b_i} into
        # def p():
        #    r = {}
        #    for v_i in e_i
        #       if b_i:
        #          r.__setitem__(c,d)
        #    return r
        # y=p()
        _empty_ast_arguments = ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return PatternRule(
            assign(value=ast_dictComp()),
            lambda term: ListEdit(
                self.fresh_names(
                    ["p", "r"],
                    lambda new_name: [
                        ast.FunctionDef(
                            name=new_name("p", "store").id,
                            args=_empty_ast_arguments,
                            body=[
                                ast.Assign(
                                    targets=[new_name("r", "store")],
                                    value=ast.Dict(keys=[], values=[]),
                                ),
                                self._nested_fors_and_ifs_of(
                                    term.value.generators,
                                    ast.Expr(
                                        ast.Call(
                                            func=ast.Attribute(
                                                value=new_name("r", "load"),
                                                attr="__setitem__",
                                                ctx=ast.Load(),
                                            ),
                                            args=[term.value.key, term.value.value],
                                            keywords=[],
                                        )
                                    ),
                                ),
                                ast.Return(new_name("r", "load")),
                            ],
                            decorator_list=[],
                            returns=None,
                            type_comment=None,
                        ),
                        ast.Assign(
                            targets=term.targets,
                            value=ast.Call(
                                func=new_name("p", "load"), args=[], keywords=[]
                            ),
                        ),
                    ],
                )
            ),
            "handle_assign_dictComp",
        )

    def _handle_assign(self) -> Rule:
        return first(
            [
                self._handle_assign_unaryop(),
                self._handle_assign_subscript_slice_all(),
                self._handle_assign_binop_left(),
                self._handle_assign_binop_right(),
                self._handle_assign_attribute(),
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
                self._handle_assign_call_empty_keyword_arg(),
                # Rules for comprehensions
                self._handle_assign_listComp(),
                self._handle_assign_setComp(),
                self._handle_assign_dictComp(),
                # Rules for dict (as a special function name)
                self._handle_assign_unary_dict(),
                self._handle_assign_binary_dict_left(),
                self._handle_assign_binary_dict_right(),
                self._handle_left_value_all(),
            ]
        )

    def _handle_boolop_binarize(self) -> Rule:
        # This rule eliminates non-binary "and" and "or" operators.
        #
        # Boolean operators -- "and" and "or" -- are not necessarily binary operators.
        # "a and b and c" is parsed as a single "and" operator with three operands!
        # This rule rewrites "a and b and c and ..." into "(a and b) and c and..."
        # If the rule is then repeated until it attains a fixpoint we attain the
        # invariant that every Boolean operator is also a binary operator.
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
        # This rule eliminates "and" and "or" operators with two operands where
        # the left operand is complex. It rewrites:
        #
        # x = complex and y
        #
        # to
        #
        # t = complex
        # x = t and y
        #
        # And similarly for "or".
        return PatternRule(
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
        # This rule entirely eliminates "and" operators with two operands where the
        # left operand is an identifier. It rewrites:
        #
        # x = id and y
        #
        # to
        #
        # if id:
        #   x = y
        # else:
        #   x = id
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
        # This rule entirely eliminates "or" operators with two operands where the
        # left operand is an identifier. It rewrites:
        #
        # x = id or y
        #
        # to
        #
        # if id:
        #   x = id
        # else:
        #   x = y
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
        # This rule eliminates all "and" and "or" operators from the program.
        return first(
            [
                self._handle_boolop_binarize(),
                self._handle_assign_boolop_linearize(),
                self._handle_assign_and2if(),
                self._handle_assign_or2if(),
            ]
        )

    def _handle_compare_binarize(self) -> Rule:
        # This rule eliminates non-binary comparison operators where the *second*
        # leftmost operand is an identifier. This could use some explanation.
        #
        # In Python the comparison operators are not necessarily binary operators.
        # An expression of the form
        #
        # x = a < b > c
        #
        # is equivalent to
        #
        # x = a < b and b > c
        #
        # Except that b is evaluated *only once*.  We therefore must ensure that
        # "b" in this case has no side effects before we can do this rewrite. We
        # rewrite:
        #
        # x = anything OP id OP anything ...
        #
        # to
        #
        # x = (anything OP id) and (id OP anything ...)
        #
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
        # This rule eliminates comparison operations where the leftmost operand
        # is not an identifier, regardless of how many operands and operators
        # there are in the operation. It rewrites:
        #
        # x = complex OP anything ...
        #
        # to
        #
        # t = complex
        # x = t OP anything ...
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
        # This rule eliminates comparison operations where the leftmost operand
        # is an identifier and the second-from-the-leftmost operand is not an
        # identifier, regardless of how many operands and operators there are in
        # the operation. It rewrites:
        #
        # x = id OP complex ...
        #
        # to
        #
        # t = complex
        # x = id OP t ...
        #
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
        # This rule simplifies the left two operands of a comparison
        # operations to be both identifiers, and breaks up non-binary
        # comparison operations by introducing an "and".
        #
        # Since we have other rules which eventually eliminate "and" expressions
        # entirely, repeated application of these rules will reach a fixpoint where
        # every comparison is a binary operator containing only identifiers.
        #
        # For example, the combination of this rule and the Boolean operator rules
        # when executed until a fixpoint is reached would rewrite:
        #
        # x = (a + b) < (c + d) < (e + f)
        #
        # to:
        #
        # t1 = a + b
        # t2 = c + d
        # t3 = t1 < t2
        # if t3:
        #   t4 = e + f
        #   x = t2 < t4
        # else:
        #   x = t3
        #
        # which has the same semantics.  Note that if (a + b) < (c + d) is false
        # then e + f is never computed.

        return first(
            [
                self._handle_compare_binarize(),
                self._handle_assign_compare_righthandside(),
                self._handle_assign_compare_lefthandside(),
            ]
        )

    # TODO: We need to find a good way to order things in this file

    def _handle_left_value_all(self) -> Rule:
        """Put the left_value of an assignment in SSA form"""
        # TODO: Add the various rewrite rules into the following list
        return first(
            [
                self._handle_left_value_attributeref(),
                self._handle_left_value_subscript_value(),
                self._handle_left_value_subscript_slice_index(),
                self._handle_left_value_subscript_slice_lower(),
                self._handle_left_value_subscript_slice_upper(),
                self._handle_left_value_subscript_slice_step(),
                self._handle_left_value_list_star(),
                self._handle_left_value_list_list(),
                self._handle_left_value_list_not_starred(),
                self._handle_left_value_list_starred(),
            ]
        )

    def _handle_left_value_attributeref(self) -> Rule:
        """Rewrites like a.b.c = z → x = a.b; x.c = z"""
        return PatternRule(
            assign(targets=[attribute(value=_not_identifier)], value=name()),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.targets[0].value,
                lambda source_term, new_name: ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id=new_name, ctx=ast.Load()),
                            attr=source_term.targets[0].attr,
                            ctx=ast.Store(),
                        )
                    ],
                    value=source_term.value,
                ),
            ),
            "handle_left_value_attributeref",
        )

    def _handle_left_value_subscript_value(self) -> Rule:
        """Rewrites like a.b[c] = z → x = a.b; x[c] = z"""
        return PatternRule(
            assign(targets=[subscript(value=_not_identifier)], value=name()),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.targets[0].value,
                lambda source_term, new_name: ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id=new_name, ctx=ast.Load()),
                            slice=source_term.targets[0].slice,
                            ctx=ast.Store(),
                        )
                    ],
                    value=source_term.value,
                ),
            ),
            "_handle_left_value_subscript_value",
        )

    def _handle_left_value_subscript_slice_index(self) -> Rule:
        """Rewrites like a[b.c] = z → x = b.c; a[x] = z"""
        return PatternRule(
            assign(
                targets=[subscript(value=name(), slice=index(value=_not_identifier))],
                value=name(),
            ),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.targets[0].slice.value,
                lambda source_term, new_name: ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=source_term.targets[0].value,
                            slice=ast.Index(
                                value=ast.Name(id=new_name, ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                            ctx=ast.Store(),
                        )
                    ],
                    value=source_term.value,
                ),
            ),
            "_handle_left_value_subscript_slice_index",
        )

    def _handle_left_value_subscript_slice_lower(self) -> Rule:
        """Rewrites like a[b.c:] = z → x = b.c; a[x:] = z."""
        return PatternRule(
            assign(
                targets=[
                    subscript(
                        value=name(), slice=slice_pattern(lower=_neither_name_nor_none)
                    )
                ],
                value=name(),
            ),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.targets[0].slice.lower,
                lambda source_term, new_name: ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=source_term.targets[0].value,
                            slice=ast.Slice(
                                lower=ast.Name(id=new_name, ctx=ast.Load()),
                                upper=source_term.targets[0].slice.upper,
                                step=source_term.targets[0].slice.step,
                            ),
                            ctx=ast.Store(),
                        )
                    ],
                    value=source_term.value,
                ),
            ),
            "_handle_left_value_subscript_slice_lower",
        )

    def _handle_left_value_subscript_slice_upper(self) -> Rule:
        """Rewrites like a[:b.c] = z → x = b.c; a[:x] = z."""
        return PatternRule(
            assign(
                targets=[
                    subscript(
                        value=name(),
                        slice=slice_pattern(
                            lower=_name_or_none, upper=_neither_name_nor_none
                        ),
                    )
                ],
                value=name(),
            ),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.targets[0].slice.upper,
                lambda source_term, new_name: ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=source_term.targets[0].value,
                            slice=ast.Slice(
                                lower=source_term.targets[0].slice.lower,
                                upper=ast.Name(id=new_name, ctx=ast.Load()),
                                step=source_term.targets[0].slice.step,
                            ),
                            ctx=ast.Store(),
                        )
                    ],
                    value=source_term.value,
                ),
            ),
            "_handle_left_value_subscript_slice_upper",
        )

    def _handle_left_value_subscript_slice_step(self) -> Rule:
        """Rewrites like a[:c:d.e] = z → x = c.d; a[b:c:x] = z."""
        return PatternRule(
            assign(
                targets=[
                    subscript(
                        value=name(),
                        slice=slice_pattern(
                            lower=_name_or_none,
                            upper=_name_or_none,
                            step=_neither_name_nor_none,
                        ),
                    )
                ],
                value=name(),
            ),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.targets[0].slice.step,
                lambda source_term, new_name: ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=source_term.targets[0].value,
                            slice=ast.Slice(
                                lower=source_term.targets[0].slice.lower,
                                upper=source_term.targets[0].slice.upper,
                                step=ast.Name(id=new_name, ctx=ast.Load()),
                            ),
                            ctx=ast.Store(),
                        )
                    ],
                    value=source_term.value,
                ),
            ),
            "_handle_left_value_subscript_slice_step",
        )

    def _handle_left_value_list_star(self) -> Rule:
        """Rewrites like [*a.b] = z → [*y] = z; a.b = y."""
        # Note: This type of rewrite should not be "generalized" to
        # have anything come after *a.b because that would change order
        # of evaluation within the pattern.

        return PatternRule(
            assign(
                targets=[ast_luple(elts=[starred(value=_not_identifier)])],
                value=name(),
            ),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.value,
                lambda source_term, new_name: ast.Assign(
                    targets=[source_term.targets[0].elts[0].value],
                    value=new_name,
                ),
                lambda source_term, new_name: [
                    ast.List(elts=[ast.Starred(value=new_name)])
                ],
            ),
            "_handle_left_value_list_star",
        )

    def _handle_left_value_list_list(self) -> Rule:
        """Rewrites like [[e]] = z → [y] = z; [e] = y."""
        # Note: This type of rewrite should not be "generalized" to
        # have anything come after [e] because that would change order
        # of evaluation within the pattern.
        return PatternRule(
            assign(
                targets=[ast_luple(elts=[ast_luple(elts=[anyPattern])])],
                value=name(),
            ),
            self._transform_with_name(
                "x",
                lambda source_term: source_term.value,
                lambda source_term, new_name: ast.Assign(
                    targets=[ast.List(elts=[source_term.targets[0].elts[0].elts[0]])],
                    value=new_name,
                ),
                lambda source_term, new_name: [ast.List(elts=[new_name])],
            ),
            "_handle_left_value_list_list_star",
        )

    def _handle_left_value_list_not_starred(self) -> Rule:
        """Rewrites like [a.b.c, d] = z → a.b.c = z[0]; [d] = z[1:]."""
        # Here we are handling lists with more than one element.
        return PatternRule(
            assign(
                targets=[
                    ast_luple(
                        elts=HeadTail(_not_starred, HeadTail(anyPattern, anyPattern))
                    )
                ],
                value=name(),
            ),
            lambda source_term: ListEdit(
                [
                    ast.Assign(
                        targets=[source_term.targets[0].elts[0]],
                        value=ast.Subscript(
                            value=source_term.value,
                            slice=ast.Index(value=ast.Num(n=0)),
                            ctx=ast.Load(),
                        ),
                    ),
                    ast.Assign(
                        targets=[
                            ast.List(
                                elts=source_term.targets[0].elts[1:], ctx=ast.Store()
                            )
                        ],
                        value=ast.Subscript(
                            value=source_term.value,
                            slice=ast.Slice(lower=ast.Num(n=1), upper=None, step=None),
                            ctx=ast.Load(),
                        ),
                    ),
                ],
            ),
            "_handle_left_value_list_not_starred",
        )

    def _handle_left_value_list_starred(self) -> Rule:
        """Rewrites like [*c, d] = z → [*c] = z[:-1]; d = z[-1]."""
        # Here we are handling lists with more than one element.
        return PatternRule(
            assign(
                targets=[
                    ast_luple(
                        elts=HeadTail(starred(), HeadTail(anyPattern, anyPattern))
                    )
                ],
                value=name(),
            ),
            lambda source_term: ListEdit(
                [
                    ast.Assign(
                        targets=[
                            ast.List(
                                elts=source_term.targets[0].elts[:-1], ctx=ast.Store()
                            )
                        ],
                        value=ast.Subscript(
                            value=source_term.value,
                            slice=ast.Slice(lower=None, upper=ast.Num(n=-1), step=None),
                            ctx=ast.Load(),
                        ),
                    ),
                    ast.Assign(
                        targets=[source_term.targets[0].elts[-1]],
                        value=ast.Subscript(
                            value=source_term.value,
                            slice=ast.Index(value=ast.Num(n=-1)),
                            ctx=ast.Load(),
                        ),
                    ),
                ],
            ),
            "_handle_left_value_list_starred",
        )

    def single_assignment(self, node: ast.AST) -> ast.AST:
        return self._rules(node).expect_success()


def single_assignment(node: ast.AST) -> ast.AST:
    s = SingleAssignment()
    return s.single_assignment(node)
