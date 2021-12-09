#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
import inspect
import sys
import types
from typing import Callable, List, Tuple, Optional, Dict, Any

import astor
from beanmachine.ppl.compiler.ast_patterns import (
    ast_if,
    ast_for,
    assign,
    ast_assert,
    ast_domain,
    attribute,
    aug_assign,
    binary_compare,
    binop,
    call,
    index,
    keyword,
    load,
    name,
    starred,
    subscript,
    slice_pattern,
    unaryop,
)
from beanmachine.ppl.compiler.internal_error import LiftedCompilationError
from beanmachine.ppl.compiler.patterns import match_any
from beanmachine.ppl.compiler.rules import (
    AllOf as all_of,
    FirstMatch as first,
    Pattern,
    PatternRule,
    Rule,
    TryOnce as once,
    remove_from_list,
    ListEdit,
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
_name_or_none = match_any(name(), None)


def _parse_expr(source: str) -> ast.expr:
    # Takes a string containing an expression; ast.parse creates
    # Module(body=[Expr(value=THE_EXPRESSION)]); obtain the expression.
    e = ast.parse(source).body[0]
    assert isinstance(e, ast.Expr)
    return e.value


_bmg = _parse_expr("bmg")


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
            [
                a.value.func,
                a.value.args[0].value,
                a.value.keywords[0].value,
            ],
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


# a = b[c] --> a = bmg.handle_index(b, c)
_handle_index = PatternRule(
    assign(value=subscript(slice=index())),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("handle_index", [a.value.value, a.value.slice.value])
    ),
)

_ast_none = ast.Constant(value=None, kind=None)


def _or_none(a):
    return _ast_none if a is None else a


# a = b[c:d:e] --> a = bmg.handle_slice(b, c, d, e)
_handle_slice = PatternRule(
    assign(
        value=subscript(
            value=name(),
            slice=slice_pattern(
                lower=_name_or_none, upper=_name_or_none, step=_name_or_none
            ),
        )
    ),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call(
            "handle_slice",
            [
                a.value.value,
                _or_none(a.value.slice.lower),
                _or_none(a.value.slice.upper),
                _or_none(a.value.slice.step),
            ],
        ),
    ),
)

# a[b] = e --> bmg.handle_subscript_assign(a, b, None, None, e)
_handle_subscript_assign_index = PatternRule(
    assign(
        targets=[
            subscript(
                value=name(),
                slice=index(value=name()),
            )
        ],
        value=name(),
    ),
    lambda a: ast.Expr(
        _make_bmg_call(
            "handle_subscript_assign",
            [
                a.targets[0].value,
                a.targets[0].slice.value,
                _ast_none,
                _ast_none,
                a.value,
            ],
        ),
    ),
)


# a[b:c:d] = e --> bmg.handle_subscript_assign(a, b, c, d, e)
_handle_subscript_assign_slice = PatternRule(
    assign(
        targets=[
            subscript(
                value=name(),
                slice=slice_pattern(
                    lower=_name_or_none,
                    upper=_name_or_none,
                    step=_name_or_none,
                ),
            )
        ],
        value=name(),
    ),
    lambda a: ast.Expr(
        _make_bmg_call(
            "handle_subscript_assign",
            [
                a.targets[0].value,
                _or_none(a.targets[0].slice.lower),
                _or_none(a.targets[0].slice.upper),
                _or_none(a.targets[0].slice.step),
                a.value,
            ],
        ),
    ),
)

_assignments_to_bmg: Rule = first(
    [
        _handle_dot,
        _handle_call,
        # Unary operators: ~ not + -
        _handle_unary(ast.Invert, "handle_invert"),
        _handle_unary(ast.Not, "handle_not"),
        _handle_unary(ast.UAdd, "handle_uadd"),
        _handle_unary(ast.USub, "handle_negate"),
        _handle_binary(ast.Add, "handle_addition"),
        # Binary operators & | / // << % * ** >> - @
        # "and" and "or" are already eliminated by the single
        # assignment rewriter.
        _handle_binary(ast.BitAnd, "handle_bitand"),
        _handle_binary(ast.BitOr, "handle_bitor"),
        _handle_binary(ast.BitXor, "handle_bitxor"),
        _handle_binary(ast.Div, "handle_division"),
        _handle_binary(ast.FloorDiv, "handle_floordiv"),
        _handle_binary(ast.LShift, "handle_lshift"),
        _handle_binary(ast.MatMult, "handle_matrix_multiplication"),
        _handle_binary(ast.Mod, "handle_mod"),
        _handle_binary(ast.Mult, "handle_multiplication"),
        _handle_binary(ast.Pow, "handle_power"),
        _handle_binary(ast.RShift, "handle_rshift"),
        _handle_binary(ast.Sub, "handle_subtraction"),
        # []
        _handle_index,
        _handle_slice,
        # Comparison operators: == != > >= < <=
        # is, is not, in, not in
        _handle_comparison(binary_compare(ast.Eq), "handle_equal"),
        _handle_comparison(binary_compare(ast.NotEq), "handle_not_equal"),
        _handle_comparison(binary_compare(ast.Gt), "handle_greater_than"),
        _handle_comparison(binary_compare(ast.GtE), "handle_greater_than_equal"),
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
        # Indexed assignments
        _handle_subscript_assign_index,
        _handle_subscript_assign_slice,
    ]
)

# Rewrite
#
# if ID:
#    consequence
# else:
#    alternative
#
# to
#
# bmg.handle_if(ID)
# if ID:
#    ...
#
# Note that handle_if must not be an operand of the top-down combinator
# because we would just enter an infinite loop of adding the handler
# before the if-statement.

_handle_if = PatternRule(
    ast_if(test=name()),
    lambda a: ListEdit(
        [
            ast.Expr(_make_bmg_call("handle_if", [a.test])),
            a,
        ]
    ),
)

_handle_for = PatternRule(
    ast_for(iter=name()),
    lambda a: ListEdit(
        [
            ast.Expr(_make_bmg_call("handle_for", [a.iter])),
            a,
        ]
    ),
)

_control_flow_to_bmg: Rule = first(
    [
        _handle_for,
        _handle_if,
    ]
)

# Note that we are NOT attempting to iterate to a fixpoint here; we do a transformation
# on every statement once.
_statements_to_bmg: Rule = all_of(
    [
        _top_down(once(_assignments_to_bmg)),
        _bottom_up(once(_control_flow_to_bmg)),
    ]
)


# TODO: Add classes, lambdas, and so on
_supported_code_containers = {types.MethodType, types.FunctionType}


def _bm_ast_to_bmg_ast(a: ast.Module) -> ast.Module:
    """This function takes any AST module, say:

        def f():
            return norm() + 1.0

    and transforms it to a form where every operation becomes a call
    into the BMG runtime:

        def f():
            t1 = []
            t2 = bmg.handle_function(norm, t1)
            t3 = 1.0
            t4 = bmg.handle_add(t2, t3)
            return t4

    It returns the AST of the transformed code as a module.
    """

    no_asserts = _eliminate_all_assertions(a).expect_success()
    assert isinstance(no_asserts, ast.Module)
    sa = single_assignment(no_asserts)
    assert isinstance(sa, ast.Module)
    # Now we're in single assignment form.
    rewrites = [_statements_to_bmg]
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


def _get_lines_ast(f: Callable) -> Tuple[str, ast.Module]:
    """Takes a function object, returns the code containing
    its definition as both text and a module. Note that if the
    function is a lambda then we return all the lines containing
    the lambda, not just the lambda."""
    lines, _ = inspect.getsourcelines(f)
    # The code may be indented because it is a local function or class member;
    # either way, we cannot parse an indented function. Unindent it.
    source = "".join(_unindent(lines))
    module = ast.parse(source)
    return source, module


def _transform_lambda(f: Callable) -> Tuple[Optional[List[ast.stmt]], str, str]:

    """Takes a lambda such as

    lambda n: norm() + 1.0

    and transforms it to a form where every operation becomes a call
    into the BMG runtime:

        def t1(n):
            t2 = [n]
            t3 = bmg.handle_function(norm, t2)
            t4 = 1.0
            t5 = bmg.handle_add(t3, t4)
            return t5
        _lambda = t1

    It returns:
    * the body of the transformed function, or None if the transformation failed
    * the name of an identifier which refers to the transformed function (_lambda in
      this example)
    * the source code of the original lambda"""

    # See http://xion.io/post/code/python-get-lambda-code.html and its comments
    # for an extended discussion of how broken python is when you need the
    # source code for a lambda.
    #
    # Summary: getsourceslines does exactly what it says on the tin: gives you
    # the source code **lines** associated with a function. If for example we
    # have the line "x = y(lambda: 2, lambda: 3)" and we wish to know the source
    # code of the first lambda, calling getsourcelines returns that string,
    # containing the entire line.
    #
    # How can we then tell which lambda is the one we want? There is no reliable way to
    # do so! The __code__ object of the function only contains the line number, not the
    # column offset.
    #
    # The vast majority of the time there will be a single lambda on the line, so we'll
    # implement for that scenario.
    #
    # TODO: Consider producing a warning or error if we cannot determine which lambda
    # is intended.

    # TODO: return None if we are unable to get the source
    source, module = _get_lines_ast(f)
    all_lambdas = [
        astnode for astnode in ast.walk(module) if isinstance(astnode, ast.Lambda)
    ]
    if len(all_lambdas) != 1:
        return None, "", ""

    name = "_lambda"
    # Give the rewriter "_lambda = lambda: whatever", and it will rewrite that statement
    # into a function definition and assignment.
    assignment = ast.Module(
        body=[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())], value=all_lambdas[0]
            ),
        ]
    )

    bmg = _bm_ast_to_bmg_ast(assignment)
    assert isinstance(bmg, ast.Module)
    assert len(bmg.body) == 2
    return bmg.body, name, source


def _transform_function(f: Callable) -> Tuple[Optional[List[ast.stmt]], str, str]:
    """Takes a function such as

        @functional
        def f():
            return norm() + 1.0

    and transforms it to a form where every operation becomes a call
    into the BMG runtime:

        def f():
            t1 = []
            t2 = bmg.handle_function(norm, t1)
            t3 = 1.0
            t4 = bmg.handle_add(t2, t3)
            return t4
        t5 = [f]
        f = bmg.handle_function(functional, t5)

    It returns:
    * a list of statement ASTs of the transformed function, or None if the
      transformation failed
    * the name of an identifier which refers to the transformed function
    * the source code of the original function

    Notice that if the original function is decorated with the @decorator syntax
    then the transformed function is decorated by calling the decorator directly.
    """

    # We need special handling for lambdas.
    if f.__name__ == "<lambda>":
        return _transform_lambda(f)

    # TODO: return None if we are unable to get the source
    source, original_ast = _get_lines_ast(f)
    assert len(original_ast.body) == 1
    if not isinstance(original_ast.body[0], ast.FunctionDef):
        return None, "", ""
    transformed_ast: ast.Module = _bm_ast_to_bmg_ast(original_ast)
    assert len(transformed_ast.body) >= 1
    funcdef = transformed_ast.body[0]
    assert isinstance(funcdef, ast.FunctionDef)
    return transformed_ast.body, funcdef.name, source


def _create_enclosing_helper(
    f: Callable,
    transformed_body: List[ast.stmt],
    name: str,
    helper_name: str,
) -> ast.AST:
    """Takes:

    * the original function being transformed
    * the AST of the transformed body
    * the name of an identifier referring to the function in the transformed code
    * the name of a helper method

    Returns the AST of a helper method which closes the transformed body over:

    * the BMG runtime that accumulates the operations
    * the free variables of the original function"""

    # For example, if we are given the transformed method
    #
    # def f():
    #     t1 = []
    #     t2 = bmg.handle_function(norm, t1)
    #     t3 = 1.0
    #     t4 = bmg.handle_add(t2, t3)
    #     return t4
    #
    # Then we generate:
    #
    # def f_helper(bmg):
    #     def f():
    #         t1 = []
    #         t2 = bmg.handle_function(norm, t1)
    #         t3 = 1.0
    #         t4 = bmg.handle_add(t2, t3)
    #         return t4
    #     return f
    #
    # Suppose we've been asked to transform a function which is closed over some outer
    # variables; how does that work?  For example:
    #
    # def x(offset):
    #   def y():
    #     return flip() + offset
    #   return y
    #
    # f = x(1)
    #
    # @functional def some_functional():
    #   return some_rv(f())  # some_rv(1) or some_rv(2)
    #
    # The *functional* never calls x; by the time we call y(), x(1) is long gone,
    # so we cannot rely on the body of x being transformed.  Somehow we must transform
    # y to:
    #
    # def y():
    #    t1 = bmg.handle_function(flip, [])
    #    t2 = bmg.handle_add(t1, offset)
    #    return t2
    #
    # where offset needs to be 1.
    #
    # Python implements closures by storing the names of the outer variables in a tuple
    # at y.__code__.co_freevars, and the cells (references to *variables*) in a tuple
    # at y.__closure__. You might think that we could simply generate the function above
    # and the set co_freevars and __closure__ on the new function object to the appropriate
    #  values, but unfortunately these are both read-only attributes of function objects.
    #
    # Instead what we'll do is generate:
    #
    # def y_helper(bmg, offset):
    #   def y():
    #     t1 = bmg.handle_function(flip, [])
    #     t2 = bmg.handle_add(t1, offset)
    #     return t2
    #   return y
    #
    # and then call y_helper with the appropriate values. That is, generate a new set of
    # identical outer variables with the same values.
    #
    # How can this go wrong?
    #
    # Closures are closed over *variables*, not *values*, and this is observable when
    # an inner function uses the *nonlocal* statement:
    #
    # def new_counter():
    #   counter = 0
    #   def inc():
    #     nonlocal counter
    #     counter += 1
    #     print(counter)
    #   def dec():
    #     nonlocal counter
    #     counter += 1
    #     print(counter)
    #   return inc, dec
    # i, d = new_counter()
    # i() # 1
    # i() # 2
    # d() # 1
    #
    # In this example the nonlocal statement causes "counter" to be an alias for the
    # outer variable.
    #
    # If we were asked to compile functions i or d, we would pass in the *current* value
    # of counter, but we would not create an alias to the *existing* counter variable that
    # i and d are closed over.
    #
    # For now, we're going to have to live with this.  It should be rare for a model
    # to have a callee that does this.
    #
    # TODO: Consider detecting jitted functions which use the nonlocal statement, and
    # producing an error or warning.
    #
    # TODO: rename bmg outer variable to something less likely
    # to be shadowed by an inner variable.

    assert type(f) in _supported_code_containers

    helper_parameters = [ast.arg(arg="bmg", annotation=None)]
    if hasattr(f, "__code__"):
        code = f.__code__  # pyre-ignore
        if hasattr(code, "co_freevars"):
            for outer in code.co_freevars:
                helper_parameters.append(ast.arg(arg=outer, annotation=None))

    helper_args = ast.arguments(
        posonlyargs=[],
        args=helper_parameters,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    helper_body = transformed_body + [
        ast.Return(value=ast.Name(id=name, ctx=ast.Load())),
    ]

    helper_func = ast.FunctionDef(
        name=helper_name,
        args=helper_args,
        body=helper_body,
        decorator_list=[],
        returns=None,
    )

    helper_ast = ast.Module(body=[helper_func], type_ignores=[])
    ast.fix_missing_locations(helper_ast)

    return helper_ast


def _get_globals(f: Callable) -> Dict:
    """Given a function, returns the global variable dictionary of the
    module containing the function."""

    if f.__module__ not in sys.modules:
        msg = (
            f"module {f.__module__} for function {f.__name__} not "
            + f"found in sys.modules.\n{str(sys.modules.keys())}"
        )
        raise Exception(msg)
    return sys.modules[f.__module__].__dict__


def _get_helper_arguments(
    original_function: Callable, runtime: BMGRuntime
) -> List[Any]:
    # The helper function takes the runtime as its first argument and
    # then all the outer variables of the original function.

    arguments = [runtime]
    if hasattr(original_function, "__closure__"):
        closure = original_function.__closure__  # pyre-ignore
        if closure is not None:
            for cell in closure:
                arguments.append(cell.cell_contents)
    return arguments


def _bm_function_to_bmg_function(
    original_function: Callable, runtime: BMGRuntime
) -> Callable:

    """Takes a function object and -- if possible -- returns a function of the same
    signature which calls the BMGRuntime object on each operation that was in the
    original function. If not possible, it returns the original function and we
    hope that it did not do anything involving a stochastic quantity.
    """

    # We only know how to transform certain kinds of code containers.
    # If we don't have one of those, just return the function unmodified
    # and hope for the best.

    if type(original_function) not in _supported_code_containers:
        return original_function

    # First obtain the transformed function itself; the resulting AST will not
    # yet be closed over either the runtime or any outer variables of the
    # original function.

    transformed_body, name, original_source = _transform_function(original_function)
    if transformed_body is None:
        return original_function

    # Now create a helper function wrapping the transformed function; this
    # wrapper creates a closure.

    helper_name = name + "_helper"
    helper_ast = _create_enclosing_helper(
        original_function, transformed_body, name, helper_name
    )

    # We now have an AST containing a function, foo_helper, which when called
    # will return the rewritten function. We need to call this helper somehow.
    # First thing to do is to compile it.
    #
    # Python requires a "filename" to compile code; we'll make up a fake one.
    filename = "<BMGJIT>"
    try:
        compiled_helper = compile(helper_ast, filename, "exec")
    except Exception as ex:
        # If something went wrong during compilation, we probably have a
        # bug in the AST rewriting step. Report the error here.
        raise LiftedCompilationError(original_source, helper_ast, ex) from ex

    # The AST is now compiled into bytecode but has not yet been executed.
    # That bytecode, when executed, will define a module containing the
    # helper function.

    # What happens if the new code accesses a global variable defined in the
    # *original* code's module?  We need to tell Python that the new module
    # and the old module have the same global variables.
    #
    # We need the dictionary mapping from global variables to their values.

    original_globals = _get_globals(original_function)

    # All right, we have the bytecode of the helper function, and we have
    # the global state of the original function. When we execute the bytecode
    # we've just compiled, that will add the helper method to the global variable
    # dictionary.

    exec(compiled_helper, original_globals)  # noqa

    # The helper function is now in the global variables. Obtain it and call it.

    helper_function = original_globals[helper_name]
    arguments = _get_helper_arguments(original_function, runtime)
    transformed_function = helper_function(*arguments)

    # For debugging purposes we'll stick some helpful information into
    # the function object.

    transformed_function.runtime = runtime
    transformed_function.original_function = original_function
    transformed_function.original_source = original_source
    transformed_function.transformed_ast = helper_ast
    transformed_function.transformed_source = astor.to_source(helper_ast)
    return transformed_function


def _bm_function_to_bmg_ast(f: Callable, helper_name: str) -> ast.AST:
    # TODO: This method is only here for testing purposes.  Get rid of it.
    transformed_body, name, _ = _transform_function(f)
    return _create_enclosing_helper(
        f, transformed_body, name, helper_name  # pyre-ignore
    )
