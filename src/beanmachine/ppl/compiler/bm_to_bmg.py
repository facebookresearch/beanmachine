#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
import inspect
import sys
import types
from typing import Callable, List, Tuple

import astor
from beanmachine.ppl.compiler.ast_patterns import (
    arguments,
    assign,
    ast_assert,
    ast_domain,
    attribute,
    aug_assign,
    binary_compare,
    binop,
    call,
    function_def,
    index,
    keyword,
    load,
    name,
    starred,
    subscript,
    unaryop,
)
from beanmachine.ppl.compiler.internal_error import LiftedCompilationError
from beanmachine.ppl.compiler.patterns import nonEmptyList
from beanmachine.ppl.compiler.rules import (
    AllOf as all_of,
    FirstMatch as first,
    Pattern,
    PatternRule,
    Rule,
    TryOnce as once,
    always_replace,
    remove_from_list,
)
from beanmachine.ppl.compiler.runtime import BMGRuntime
from beanmachine.ppl.compiler.single_assignment import single_assignment


# TODO: Detect unsupported operators
# TODO: Detect unsupported control flow
# TODO: Would be helpful if we could track original source code locations.

_top_down = ast_domain.top_down
_bottom_up = ast_domain.bottom_up
_descend_until = ast_domain.descend_until
_specific_child = ast_domain.specific_child

_eliminate_assertion = PatternRule(ast_assert(), lambda a: remove_from_list)

_eliminate_all_assertions: Rule = _top_down(once(_eliminate_assertion))

_bmg = ast.Name(id="bmg", ctx=ast.Load())


def _make_bmg_call(name: str, args: List[ast.AST]) -> ast.AST:
    return ast.Call(
        func=ast.Attribute(value=_bmg, attr=name, ctx=ast.Load()),
        args=args,
        keywords=[],
    )


#
# After rewriting into single assignment form, every call has one of these forms:
#
# x = f(*args, **kwargs)
# x = dict()
# x = dict(key = id)
# x = dict(**id, **id)
#
# The first is the only one we need to rewrite. We rewrite it to
#
# x = bmg.handle_function(f, args, kwargs)

_starred_id = starred(value=name())
_double_starred_id = keyword(arg=None, value=name())

_handle_call: PatternRule = PatternRule(
    assign(value=call(args=[_starred_id], keywords=[_double_starred_id])),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call(
            "handle_function",
            [a.value.func, a.value.args[0].value, a.value.keywords[0].value,],
        ),
    ),
)

_handle_dot = PatternRule(
    assign(value=attribute(ctx=load)),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call("handle_dot_get", [a.value.value, ast.Str(a.value.attr)]),
    ),
)


def _handle_unary(p: Pattern, s: str) -> PatternRule:
    # A rule which transforms
    #
    # x = OP y
    #
    # into
    #
    # x = bmg.handle_OP(y)

    return PatternRule(
        assign(value=unaryop(op=p)),
        lambda a: ast.Assign(a.targets, _make_bmg_call(s, [a.value.operand])),
    )


def _handle_binary(p: Pattern, s: str) -> PatternRule:
    # A rule which transforms
    #
    # x = y OP z
    #
    # into
    #
    # x = bmg.handle_OP(y, z)

    return PatternRule(
        assign(value=binop(op=p)),
        lambda a: ast.Assign(
            a.targets, _make_bmg_call(s, [a.value.left, a.value.right])
        ),
    )


def _handle_aug_assign(op: Pattern, s: str) -> PatternRule:
    # A rule which transforms
    #
    # x OP= y
    #
    # into
    #
    # x = bmg.handle_iOP(x, y)
    #
    # Note that the x on the left of the assignment must be a Store()
    # and the one on the right must be a Load().

    return PatternRule(
        aug_assign(target=name(), value=name(), op=op),
        lambda a: ast.Assign(
            [a.target],
            _make_bmg_call(s, [ast.Name(id=a.target.id, ctx=ast.Load()), a.value]),
        ),
    )


def _handle_comparison(p: Pattern, s: str) -> PatternRule:
    # A rule which transforms
    #
    # x = y OP z
    #
    # into
    #
    # x = bmg.handle_OP(y, z)

    return PatternRule(
        assign(value=p),
        lambda a: ast.Assign(
            a.targets, _make_bmg_call(s, [a.value.left, a.value.comparators[0]])
        ),
    )


_handle_index = PatternRule(
    assign(value=subscript(slice=index())),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("handle_index", [a.value.value, a.value.slice.value])
    ),
)

_math_to_bmg: Rule = _top_down(
    once(
        first(
            [
                _handle_dot,
                _handle_call,
                # Unary operators: ~ not + -
                _handle_unary(ast.Invert, "handle_invert"),
                _handle_unary(ast.Not, "handle_not"),
                _handle_unary(ast.UAdd, "handle_uadd"),
                _handle_unary(ast.USub, "handle_negate"),
                _handle_binary(ast.Add, "handle_addition"),
                # Binary operators & | / // << % * ** >> -
                # TODO: @ (matrix multiplication)
                # "and" and "or" are already eliminated by the single
                # assignment rewriter.
                _handle_binary(ast.BitAnd, "handle_bitand"),
                _handle_binary(ast.BitOr, "handle_bitor"),
                _handle_binary(ast.BitXor, "handle_bitxor"),
                _handle_binary(ast.Div, "handle_division"),
                _handle_binary(ast.FloorDiv, "handle_floordiv"),
                _handle_binary(ast.LShift, "handle_lshift"),
                _handle_binary(ast.Mod, "handle_mod"),
                _handle_binary(ast.Mult, "handle_multiplication"),
                _handle_binary(ast.Pow, "handle_power"),
                _handle_binary(ast.RShift, "handle_rshift"),
                _handle_binary(ast.Sub, "handle_subtraction"),
                # []
                _handle_index,
                # Comparison operators: == != > >= < <=
                # is, is not, in, not in
                _handle_comparison(binary_compare(ast.Eq), "handle_equal"),
                _handle_comparison(binary_compare(ast.NotEq), "handle_not_equal"),
                _handle_comparison(binary_compare(ast.Gt), "handle_greater_than"),
                _handle_comparison(
                    binary_compare(ast.GtE), "handle_greater_than_equal"
                ),
                _handle_comparison(binary_compare(ast.Lt), "handle_less_than"),
                _handle_comparison(binary_compare(ast.LtE), "handle_less_than_equal"),
                _handle_comparison(binary_compare(ast.Is), "handle_is"),
                _handle_comparison(binary_compare(ast.IsNot), "handle_is_not"),
                _handle_comparison(binary_compare(ast.In), "handle_in"),
                _handle_comparison(binary_compare(ast.NotIn), "handle_not_in"),
                # Augmented assignments
                _handle_aug_assign(ast.Add, "handle_iadd"),
                _handle_aug_assign(ast.Sub, "handle_isub"),
                _handle_aug_assign(ast.Mult, "handle_imul"),
                _handle_aug_assign(ast.Div, "handle_idiv"),
                _handle_aug_assign(ast.FloorDiv, "handle_ifloordiv"),
                _handle_aug_assign(ast.Mod, "handle_imod"),
                _handle_aug_assign(ast.Pow, "handle_ipow"),
                _handle_aug_assign(ast.MatMult, "handle_imatmul"),
                _handle_aug_assign(ast.LShift, "handle_ilshift"),
                _handle_aug_assign(ast.RShift, "handle_irshift"),
                _handle_aug_assign(ast.BitAnd, "handle_iand"),
                _handle_aug_assign(ast.BitXor, "handle_ixor"),
                _handle_aug_assign(ast.BitOr, "handle_ior"),
            ]
        )
    )
)

_no_params: PatternRule = PatternRule(function_def(args=arguments(args=[])))

_replace_with_empty_list = always_replace([])

_remove_all_decorators: Rule = _descend_until(
    PatternRule(function_def(decorator_list=nonEmptyList)),
    _specific_child("decorator_list", _replace_with_empty_list,),
)


# TODO: Add classes, lambdas, and so on
_supported_code_containers = {types.MethodType, types.FunctionType}


def _bm_ast_to_bmg_ast(a: ast.AST) -> ast.AST:
    no_asserts = _eliminate_all_assertions(a).expect_success()
    assert isinstance(no_asserts, ast.Module)
    sa = single_assignment(no_asserts)
    assert isinstance(sa, ast.Module)
    # Now we're in single assignment form.
    rewrites = [_math_to_bmg, _remove_all_decorators]
    bmg = all_of(rewrites)(sa).expect_success()
    assert isinstance(bmg, ast.Module)
    return bmg


def _unindent(lines):
    # TODO: Handle the situation if there are tabs
    if len(lines) == 0:
        return lines
    num_spaces = len(lines[0]) - len(lines[0].lstrip(" "))
    if num_spaces == 0:
        return lines
    spaces = lines[0][0:num_spaces]
    return [(line[num_spaces:] if line.startswith(spaces) else line) for line in lines]


def _bm_function_to_bmg_ast(f: Callable, helper_name: str) -> Tuple[ast.AST, str]:
    """This function takes a function such as

        @random_variable
        def coin():
            return Beta(1, 2)

    and transforms it to

        def coin_helper(bmg):
            def coin():
                t1 = 1
                t2 = 2
                t3 = [t1, t2]
                t4 = bmg.handle_function(Beta, t3)
                return t4
            return coin"""

    assert type(f) in _supported_code_containers

    # TODO: f.__class__ must be 'function' or 'method'
    # TODO: Verify that we can get the source, handle it appropriately if we cannot.
    # TODO: Verify that function is not closed over any local variables
    lines, line_num = inspect.getsourcelines(f)
    # The function may be indented because it is a local function or class member;
    # either way, we cannot parse an indented function. Unindent it.
    source = "".join(_unindent(lines))
    a: ast.Module = ast.parse(source)
    assert len(a.body) == 1
    assert isinstance(a.body[0], ast.FunctionDef), f"{str(type(a.body[0]))}\n{source}"
    # TODO: Add support for classes, generators, lambdas, and so on.

    bmg = _bm_ast_to_bmg_ast(a)
    assert isinstance(bmg, ast.Module)
    assert len(bmg.body) == 1
    bmg_f = bmg.body[0]
    assert isinstance(bmg_f, ast.FunctionDef)
    name = bmg_f.name
    helper_arg = ast.arg(arg="bmg", annotation=None)
    helper_args = ast.arguments(
        posonlyargs=[],
        args=[helper_arg],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    helper_body = [
        bmg.body[0],
        ast.Return(value=ast.Name(id=name, ctx=ast.Load())),
    ]

    helper_func = ast.FunctionDef(
        name=helper_name,
        args=helper_args,
        body=helper_body,
        decorator_list=[],
        returns=None,
    )

    helper = ast.Module(body=[helper_func], type_ignores=[])
    ast.fix_missing_locations(helper)

    return helper, source


def _bm_function_to_bmg_function(f: Callable, bmg: BMGRuntime) -> Callable:
    # We only know how to compile certain kinds of code containers.
    # If we don't have one of those, just return the function unmodified
    # and hope for the best.
    if type(f) not in _supported_code_containers:
        return f

    helper_name = f.__name__ + "_helper"
    a, source = _bm_function_to_bmg_ast(f, helper_name)
    filename = "BMGJIT"

    try:
        c = compile(a, filename, "exec")
    except Exception as ex:
        raise LiftedCompilationError(source, a, ex) from ex

    mods = sys.modules.copy()
    if f.__module__ not in mods:
        msg = (
            f"module {f.__module__} for function {f.__name__} not "
            + f"found in sys.modules.\n{str(sys.modules.keys())}"
        )
        raise Exception(msg)

    g = sys.modules[f.__module__].__dict__
    exec(c, g)  # noqa
    # For debugging purposes we'll stick some helpful information into
    # the function object.
    transformed = g[helper_name](bmg)
    transformed.graph_builder = bmg
    transformed.original = f
    transformed.transformed_ast = a
    transformed.transformed_source = astor.to_source(a)
    return transformed
