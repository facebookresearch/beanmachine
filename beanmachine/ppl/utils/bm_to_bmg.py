#!/usr/bin/env python3
"""Tools to transform Bean Machine programs to Bean Machine Graph"""

import ast
import sys
import types
from typing import List

import astor
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
    function_def,
    load,
    name,
    unaryop,
)
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
from beanmachine.ppl.utils.fold_constants import _fold_unary_op, fold
from beanmachine.ppl.utils.optimize import optimize
from beanmachine.ppl.utils.patterns import ListAny
from beanmachine.ppl.utils.rules import (
    AllListMembers,
    AllOf as all_of,
    FirstMatch as first,
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
# TODO: Add support for query methods -- that is, methods that represent an
# TODO: operation on a distribution.

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


# TODO: Keywords
_handle_call = PatternRule(
    assign(value=call()),
    lambda a: ast.Assign(
        a.targets,
        _make_bmg_call(
            "handle_function",
            [a.value.func, ast.List(elts=a.value.args, ctx=ast.Load())],
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

_handle_sample = PatternRule(
    ast_return(), lambda r: ast.Return(value=_make_bmg_call("handle_sample", [r.value]))
)

# TODO: add_observation

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
            ]
        )
    )
)


_is_sample: PatternRule = PatternRule(
    function_def(decorator_list=ListAny(name(id="sample")))
)

_no_params: PatternRule = PatternRule(function_def(args=arguments(args=[])))

_returns_to_bmg: Rule = _descend_until(_is_sample, _top_down(once(_handle_sample)))

_sample_to_memoize: Rule = _descend_until(
    _is_sample,
    _specific_child(
        "decorator_list",
        SomeListMembers(
            PatternRule(
                name(id="sample"), lambda n: ast.Name(id="memoize", ctx=ast.Load())
            )
        ),
    ),
)

_header: ast.Module = ast.parse(
    """
from beanmachine.ppl.utils.memoize import memoize
from beanmachine.ppl.utils.bm_graph_builder import BMGraphBuilder
_lifted_to_bmg : bool = True
bmg = BMGraphBuilder()"""
)


def _prepend_statements(module: ast.Module, statements: List[ast.stmt]) -> ast.Module:
    return ast.Module(body=statements + module.body)


def _append_statements(module: ast.Module, statements: List[ast.stmt]) -> ast.Module:
    return ast.Module(body=module.body + statements)


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

_to_bmg = all_of([_math_to_bmg, _returns_to_bmg, _sample_to_memoize])


def _bm_to_bmg_ast(source: str) -> ast.AST:
    a: ast.Module = ast.parse(source)
    a = _eliminate_all_assertions(a).expect_success()
    # TODO: We might want to iterate the folder and optimizer until
    # TODO: they reach a fixpoint; it's possible that the optimizer
    # TODO: could someday produce a new opportunity for folding.
    f = fold(a)
    o = optimize(f)
    assert isinstance(o, ast.Module)
    # The AST has now had constants folded and associative
    # operators are nested to the left.
    es = _fix_arithmetic(o).expect_success()
    assert isinstance(es, ast.Module)
    # The AST has now eliminated all subtractions; negative constants
    # are represented as constants, not as USubs
    sa = single_assignment(es)
    assert isinstance(sa, ast.Module)
    # Now we're in single assignment form.
    bmg = _to_bmg(sa).expect_success()
    assert isinstance(bmg, ast.Module)
    bmg = _prepend_statements(bmg, _header.body)
    assert isinstance(bmg, ast.Module)
    footer: List[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="roots", ctx=ast.Store())],
            value=ast.List(
                elts=_samples_to_calls(a.body).expect_success(), ctx=ast.Load()
            ),
        ),
        ast.Expr(
            value=_make_bmg_call(
                name="remove_orphans", args=[ast.Name(id="roots", ctx=ast.Load())]
            )
        ),
    ]

    bmg = _append_statements(bmg, footer)
    ast.fix_missing_locations(bmg)
    # TODO: Fix negative constants back to standard form.
    return bmg


def to_python_raw(source: str) -> str:
    bmg: ast.AST = _bm_to_bmg_ast(source)
    p: str = astor.to_source(bmg)
    return p


def to_graph_builder(source: str) -> BMGraphBuilder:
    # TODO: Make the name unique so that if this happens more than
    # TODO: once, we're not overwriting existing work.
    filename = "<BMGAST>"
    a: ast.AST = _bm_to_bmg_ast(source)
    compiled = compile(a, filename, "exec")
    new_module = types.ModuleType(filename)
    sys.modules[filename] = new_module
    mod_globals = new_module.__dict__
    exec(compiled, mod_globals)
    bmg = mod_globals["bmg"]
    return bmg


def to_python(source: str) -> str:
    return to_graph_builder(source).to_python()


def to_cpp(source: str) -> str:
    return to_graph_builder(source).to_cpp()


def to_dot(source: str) -> str:
    return to_graph_builder(source).to_dot()


def to_bmg(source: str):
    return to_graph_builder(source).to_bmg()
