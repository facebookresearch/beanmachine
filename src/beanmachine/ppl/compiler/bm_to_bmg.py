#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
import inspect
import sys
import types
from typing import Any, Callable, Dict, List

import astor
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.compiler.internal_error import LiftedCompilationError
from beanmachine.ppl.utils.ast_patterns import (
    arguments,
    assign,
    ast_assert,
    ast_domain,
    ast_return,
    attribute,
    binop,
    call,
    constant_numeric,
    equal,
    function_def,
    greater_than,
    greater_than_equal,
    index,
    keyword,
    less_than,
    less_than_equal,
    load,
    name,
    not_equal,
    starred,
    subscript,
    unaryop,
)
from beanmachine.ppl.utils.fold_constants import _fold_unary_op, fold
from beanmachine.ppl.utils.optimize import optimize
from beanmachine.ppl.utils.patterns import ListAny, match_any
from beanmachine.ppl.utils.rules import (
    AllListMembers,
    AllOf as all_of,
    FirstMatch as first,
    ListEdit,
    PatternRule,
    Rule,
    SomeListMembers,
    TryMany as many,
    TryOnce as once,
    if_then,
    projection_rule,
    remove_from_list,
)
from beanmachine.ppl.utils.single_assignment import single_assignment


# TODO: Detect unsupported operators
# TODO: Detect unsupported control flow
# TODO: Would be helpful if we could track original source code locations.
# TODO: Collapse adds and multiplies in the graph
# TODO: Impose a restruction that a sample method always returns a distribution

_top_down = ast_domain.top_down
_bottom_up = ast_domain.bottom_up
_descend_until = ast_domain.descend_until
_specific_child = ast_domain.specific_child


_eliminate_assertion = PatternRule(ast_assert(), lambda a: remove_from_list)

_eliminate_all_assertions: Rule = _top_down(once(_eliminate_assertion))

_eliminate_subtraction = PatternRule(
    binop(op=ast.Sub),
    lambda b: ast.BinOp(
        left=b.left, op=ast.Add(), right=ast.UnaryOp(op=ast.USub(), operand=b.right)
    ),
)

_fix_usub_usub = PatternRule(
    unaryop(op=ast.USub, operand=unaryop(op=ast.USub)), lambda u: u.operand.operand
)

_fold_usub_const = PatternRule(
    unaryop(op=ast.USub, operand=constant_numeric), _fold_unary_op
)


_fix_arithmetic = all_of(
    [
        _top_down(once(_eliminate_subtraction)),
        _bottom_up(many(first([_fix_usub_usub, _fold_usub_const]))),
    ]
)


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

_handle_not = PatternRule(
    assign(value=unaryop(op=ast.Not)),
    lambda a: ast.Assign(a.targets, _make_bmg_call("handle_not", [a.value.operand])),
)

_handle_negate_usub = PatternRule(
    assign(value=unaryop(op=ast.USub)),
    lambda a: ast.Assign(a.targets, _make_bmg_call("handle_negate", [a.value.operand])),
)

_handle_addition = PatternRule(
    assign(value=binop(op=ast.Add)),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("handle_addition", [a.value.left, a.value.right])
    ),
)

_handle_multiplication = PatternRule(
    assign(value=binop(op=ast.Mult)),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call("handle_multiplication", [a.value.left, a.value.right]),
    ),
)

_handle_division = PatternRule(
    assign(value=binop(op=ast.Div)),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("handle_division", [a.value.left, a.value.right])
    ),
)

_handle_power = PatternRule(
    assign(value=binop(op=ast.Pow)),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("handle_power", [a.value.left, a.value.right])
    ),
)

_handle_equal = PatternRule(
    assign(value=equal()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call("handle_equal", [a.value.left, a.value.comparators[0]]),
    ),
)

_handle_not_equal = PatternRule(
    assign(value=not_equal()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call("handle_not_equal", [a.value.left, a.value.comparators[0]]),
    ),
)


_handle_greater_than = PatternRule(
    assign(value=greater_than()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call("handle_greater_than", [a.value.left, a.value.comparators[0]]),
    ),
)

_handle_greater_than_equal = PatternRule(
    assign(value=greater_than_equal()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call(
            "handle_greater_than_equal", [a.value.left, a.value.comparators[0]]
        ),
    ),
)

_handle_less_than = PatternRule(
    assign(value=less_than()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call("handle_less_than", [a.value.left, a.value.comparators[0]]),
    ),
)

_handle_less_than_equal = PatternRule(
    assign(value=less_than_equal()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call(
            "handle_less_than_equal", [a.value.left, a.value.comparators[0]]
        ),
    ),
)


_handle_index = PatternRule(
    assign(value=subscript(slice=index())),
    lambda a: ast.Assign(
        a.targets, _make_bmg_call("handle_index", [a.value.value, a.value.slice.value])
    ),
)


_handle_sample = PatternRule(
    ast_return(), lambda r: ast.Return(value=_make_bmg_call("handle_sample", [r.value]))
)

_math_to_bmg: Rule = _top_down(
    once(
        first(
            [
                _handle_dot,
                _handle_call,
                _handle_not,
                _handle_negate_usub,
                _handle_addition,
                _handle_multiplication,
                _handle_division,
                _handle_power,
                _handle_index,
                _handle_equal,
                _handle_not_equal,
                _handle_greater_than,
                _handle_greater_than_equal,
                _handle_less_than,
                _handle_less_than_equal,
            ]
        )
    )
)


_is_sample: PatternRule = PatternRule(
    function_def(
        decorator_list=ListAny(
            match_any(attribute(attr="random_variable"), name(id="sample"))
        )
    )
)

_is_query: PatternRule = PatternRule(
    function_def(
        decorator_list=ListAny(
            match_any(attribute(attr="functional"), name(id="query"))
        )
    )
)

_no_params: PatternRule = PatternRule(function_def(args=arguments(args=[])))

_sample_returns: Rule = _descend_until(_is_sample, _top_down(once(_handle_sample)))

_remove_query_decorator: Rule = _descend_until(
    _is_query,
    _specific_child(
        "decorator_list",
        SomeListMembers(
            PatternRule(
                match_any(attribute(attr="functional"), name(id="query")),
                lambda n: remove_from_list,
            )
        ),
    ),
)

_sample_to_memoize: Rule = _descend_until(
    _is_sample,
    _specific_child(
        "decorator_list",
        SomeListMembers(
            PatternRule(
                match_any(attribute(attr="random_variable"), name(id="sample")),
                lambda n: ListEdit(
                    [
                        ast.Call(
                            func=ast.Name(id="probabilistic", ctx=ast.Load()),
                            args=[ast.Name(id="bmg", ctx=ast.Load())],
                            keywords=[],
                            returns=None,
                        ),
                        ast.Name(id="memoize", ctx=ast.Load()),
                    ]
                ),
            )
        ),
    ),
)

_header: ast.Module = ast.parse(
    """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic
from beanmachine.ppl.compiler.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg : bool = True
bmg = BMGraphBuilder()"""
)

_short_header: ast.Module = ast.parse(
    """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.probabilistic import probabilistic"""
)


def _prepend_statements(module: ast.Module, statements: List[ast.stmt]) -> ast.Module:
    return ast.Module(body=statements + module.body, type_ignores=[])


def _append_statements(module: ast.Module, statements: List[ast.stmt]) -> ast.Module:
    return ast.Module(body=module.body + statements, type_ignores=[])


_samples_to_calls = AllListMembers(
    if_then(
        all_of([_is_sample, _no_params]),
        projection_rule(
            lambda f: ast.Call(
                func=ast.Name(id=f.name, ctx=ast.Load()), args=[], keywords=[]
            )
        ),
        projection_rule(lambda f: remove_from_list),
    )
)

_to_bmg = all_of(
    [
        _math_to_bmg,
        _sample_returns,
        _sample_to_memoize,
        _remove_query_decorator,
    ]
)


def _bm_ast_to_bmg_ast(a: ast.AST, run_optimizer: bool) -> ast.AST:
    no_asserts = _eliminate_all_assertions(a).expect_success()
    # TODO: Eventually remove the folder / optimizer; we can do optimization
    # TODO: and folding when we generate the graph. No need to do it on source.
    optimized = optimize(fold(no_asserts)) if run_optimizer else no_asserts
    assert isinstance(optimized, ast.Module)
    # The AST has now had constants folded and associative
    # operators are nested to the left.
    arithmetic_fixed = _fix_arithmetic(optimized).expect_success()
    assert isinstance(arithmetic_fixed, ast.Module)
    # The AST has now eliminated all subtractions; negative constants
    # are represented as constants, not as USubs
    sa = single_assignment(arithmetic_fixed)
    assert isinstance(sa, ast.Module)
    # Now we're in single assignment form.
    bmg = _to_bmg(sa).expect_success()
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


def _bm_function_to_bmg_ast(f: Callable, helper_name: str) -> ast.AST:
    """This function takes a function such as

        @random_variable
        def coin():
            return Beta(1, 2)

    and transforms it to

        def coin_helper(bmg):
            @probabilistic
            @memoize
            def coin():
                t1 = 1
                t2 = 2
                t3 = [t1, t2]
                t4 = bmg.handle_function(Beta, t3)
                t5 = bmg.handle_sample(t4)
                return t5
            return coin"""

    # TODO: f.__class__ must be 'function' or 'method'
    # TODO: Verify that we can get the source, handle it appropriately if we cannot.
    # TODO: Verify that function is not closed over any local variables
    lines, line_num = inspect.getsourcelines(f)
    # The function may be indented because it is a local function or class member;
    # either way, we cannot parse an indented function. Unindent it.
    source = "".join(_unindent(lines))
    a: ast.Module = ast.parse(source)
    assert len(a.body) == 1
    # TODO: What if it is an async function? Give an appropriate error.
    # TODO: Similarly for generators, lambdas, coroutines

    assert isinstance(a.body[0], ast.FunctionDef)
    bmg = _bm_ast_to_bmg_ast(a, False)
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

    # TODO: Eliminate the need to do these imports?

    helper_body = _short_header.body + [
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

    return helper


def _bm_function_to_bmg_function(f: Callable, bmg: BMGraphBuilder) -> Callable:
    helper_name = f.__name__ + "_helper"
    a = _bm_function_to_bmg_ast(f, helper_name)
    filename = "<BMGJIT>"
    # TODO: Put this in a try-except and raise a lifted compilation error
    # TODO: if there is a failure.
    c = compile(a, filename, "exec")
    if f.__module__ not in sys.modules:
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


def _bm_module_to_bmg_ast(source: str) -> ast.AST:
    a: ast.Module = ast.parse(source)
    bmg = _bm_ast_to_bmg_ast(a, True)
    assert isinstance(bmg, ast.Module)
    bmg = _prepend_statements(bmg, _header.body)
    assert isinstance(bmg, ast.Module)
    footer: List[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="roots", ctx=ast.Store())],
            value=ast.List(
                elts=_samples_to_calls(a.body).expect_success(), ctx=ast.Load()
            ),
        )
    ]

    bmg = _append_statements(bmg, footer)
    ast.fix_missing_locations(bmg)
    # TODO: Fix negative constants back to standard form.
    return bmg


def to_python_raw(source: str) -> str:
    bmg: ast.AST = _bm_module_to_bmg_ast(source)
    p: str = astor.to_source(bmg)
    return p


# Transform a model, compile the transformed state
# execute the resulting program, and return the global
# module.
def _execute(source: str) -> Dict[str, Any]:
    # TODO: Make the name unique so that if this happens more than
    # TODO: once, we're not overwriting existing work.
    filename = "<BMGAST>"
    a: ast.AST = _bm_module_to_bmg_ast(source)
    try:
        compiled = compile(a, filename, "exec")
    except Exception as ex:
        raise LiftedCompilationError(source, a, ex) from ex
    new_module = types.ModuleType(filename)
    sys.modules[filename] = new_module
    mod_globals = new_module.__dict__
    exec(compiled, mod_globals)  # noqa
    return mod_globals


def to_graph_builder(source: str) -> BMGraphBuilder:
    return _execute(source)["bmg"]


def to_python(source: str) -> str:
    return to_graph_builder(source).to_python()


def to_cpp(source: str) -> str:
    return to_graph_builder(source).to_cpp()


def to_dot(
    source: str,
    graph_types: bool = False,
    inf_types: bool = False,
    edge_requirements: bool = False,
    point_at_input: bool = False,
    after_transform: bool = False,
) -> str:
    return to_graph_builder(source).to_dot(
        graph_types, inf_types, edge_requirements, point_at_input, after_transform
    )


def to_bmg(source: str):
    return to_graph_builder(source).to_bmg()


def infer(source: str, num_samples: int = 1000) -> List[Any]:
    # TODO: Remove this API
    mod_globals = _execute(source)
    bmg = mod_globals["bmg"]
    observations = mod_globals["observations"] if "observations" in mod_globals else {}
    queries = mod_globals["queries"] if "queries" in mod_globals else []
    return bmg.infer_deprecated(queries, observations, num_samples)
