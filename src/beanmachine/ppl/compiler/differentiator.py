import _ast
import ast
import inspect
import sys
import typing

from typing import List

import beanmachine.ppl.compiler.single_assignment


def _get_globals(f: typing.Callable) -> typing.Dict:
    """Given a function, returns the global variable dictionary of the
    module containing the function."""

    if f.__module__ not in sys.modules:
        msg = (
                f"module {f.__module__} for function {f.__name__} not "
                + f"found in sys.modules.\n{str(sys.modules.keys())}"
        )
        raise Exception(msg)
    return sys.modules[f.__module__].__dict__


# at first, we will merely copy the signature and return the first parameter note that this assumes a single
# parametered primal since only one tangent is added as input rather than N for N parameters
def build_pushforward(function_def: _ast.FunctionDef) -> typing.Callable:
    single_assignment = beanmachine.ppl.compiler.single_assignment.SingleAssignment()
    single_assignment.single_assignment(function_def)
    params = []
    original_args = function_def.args
    # collect the active variables
    active_variables: List[str] = []
    pf_statements: List[_ast.AST] = []
    init_tangent = 'v'
    tangents = {}
    if isinstance(original_args, ast.arguments):
        original_args_typed: ast.arguments = original_args
        for arg in original_args_typed.args:
            new_param = _ast.arg()
            new_param.arg = arg.arg
            ann = arg.annotation
            if isinstance(ann, _ast.Name):
                new_param.annotation = _ast.Name(id=ann.id, ctx=ann.ctx)
                params.append(new_param)
                if ann.id == 'float':
                    active_variables.append(new_param.arg)
                    tangents[new_param.arg] = init_tangent

    # the final argument should be a float. It is the velocity vector argument
    params.append(_ast.arg(arg=init_tangent, annotation=_ast.Name(id='float', ctx=_ast.Load())))
    pf_args = ast.arguments(
        posonlyargs=[],
        args=params,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    for s in function_def.body:
        if isinstance(s, _ast.Assign):
            if hasattr(s, 'targets') and hasattr(s, 'value'):
                variable = s.targets[0]
                if isinstance(variable, _ast.Name):
                    op_expression = s.value
                    name = variable.id
                    if isinstance(op_expression, _ast.BinOp):
                        should_generate_tangent = False
                        for a in [op_expression.left.id, op_expression.right.id]:
                            if active_variables.__contains__(a):
                                should_generate_tangent = True
                                active_variables.append(name)
                                break
                        pf_statements.append(s)
                        if should_generate_tangent:
                            op = op_expression.op
                            if isinstance(op, _ast.Mult):
                                left = _ast.BinOp(left=_ast.Name(id=tangents[op_expression.left.id], ctx=_ast.Load()),
                                                  op=_ast.Mult(),
                                                  right=_ast.Name(op_expression.right.id, ctx=_ast.Load()))
                                right = _ast.BinOp(left=_ast.Name(id=tangents[op_expression.right.id], ctx=_ast.Load()),
                                                   op=_ast.Mult(),
                                                   right=_ast.Name(id=op_expression.left.id, ctx=_ast.Load()))
                                tangent = _ast.BinOp(left=left, op=_ast.Add(), right=right)
                            if isinstance(op, _ast.Add):
                                tangent = _ast.BinOp(
                                    left=_ast.Name(id=tangents[op_expression.left.id], ctx=_ast.Load()), op=_ast.Add(),
                                    right=_ast.Name(id=tangents[op_expression.right.id], ctx=_ast.Load()))
                            tangent_statement_name = 't_' + name
                            tangent_statement = _ast.Assign(
                                targets=[_ast.Name(id=tangent_statement_name, ctx=_ast.Store())], value=tangent)
                            tangents[name] = tangent_statement_name
                            pf_statements.append(tangent_statement)
                    elif isinstance(op_expression, _ast.UnaryOp):
                        # not implemented
                        assert 0

    # return target
    returned_primal = function_def.body[len(function_def.body) - 1]
    if isinstance(returned_primal, _ast.Return):
        p = returned_primal.value.id
        tangent = tangents[p]
        return_target = _ast.Name(id=tangent, ctx=_ast.Load())
        pf_statements.append(_ast.Return(value=return_target))
    else:
        assert 0

    # return type
    return_type = ast.Name(id=function_def.returns.id, ctx=ast.Load())

    # create a stack of the nodes to track context. This is a stack of dictionaries. In fact, it might make more
    # sense to have an object because we may reference variables from parent scopes
    finished_product = _ast.FunctionDef(name=function_def.name + "PF", args=pf_args, body=pf_statements,
                                        decorator_list=[],
                                        returns=return_type)

    # create a module to house the ast and compile to bytecode
    module = _ast.Module(body=[finished_product], type_ignores=[])
    ast.fix_missing_locations(module)
    original_globals = _get_globals(function_def)
    compiled_function = compile(module, '<string>', 'exec')
    exec(compiled_function, original_globals)
    pf = original_globals[finished_product.name]
    return pf


def primal_and_derivative(function: typing.Callable[[float], float], value: float) -> float:
    # TODO: check to see if we have already compiled a pushforward
    # get the source code
    lines, starting_line = inspect.getsourcelines(function)
    source = "".join(lines)
    module = ast.parse(source, mode='exec')
    stmts: [_ast.AST] = module.body
    if len(stmts) == 1:
        function_def_maybe = stmts[0]
        if isinstance(function_def_maybe, _ast.FunctionDef):
            function_def: _ast.FunctionDef = function_def_maybe
            pf = build_pushforward(function_def)
            return pf(value, 1.0)
    return 0
