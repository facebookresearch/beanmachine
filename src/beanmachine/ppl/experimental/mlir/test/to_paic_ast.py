import _ast
import ast
import inspect

import paic_mlir
import typing

import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils
import beanmachine.ppl.compiler.runtime


class MLIRCompileError(Exception):
    pass

class paic_ast_generator:
    def __init__(self):
        self.floatType = paic_mlir.Type("float")
        self.type_map = {'float': self.floatType }

    def python_ast_to_paic_ast(self, function_def: _ast.FunctionDef, globals:typing.Dict) -> paic_mlir.PythonFunction:
        param_list = paic_mlir.ParamList()
        symbols: typing.Dict[str, paic_mlir.DeclareValNode] = {}

        # validate function
        if isinstance(function_def, _ast.FunctionDef):
            for a in function_def.args.args:
                if isinstance(a, _ast.arg):
                    if isinstance(a.annotation, _ast.Name):
                        if not self.type_map.__contains__(a.annotation.id):
                            raise MLIRCompileError("all arguments must have types in order to be translated into llvm ir")
                        param_node = paic_mlir.ParamNode(paic_mlir.Location(0, 0), a.arg, self.type_map[a.annotation.id])
                        param_list.push_back(param_node)
                        symbols[a.arg] = param_node

            if isinstance(function_def.returns, _ast.Name):
                return_type = function_def.returns.id
                if not self.type_map.__contains__(return_type):
                    raise MLIRCompileError("return types must have types in order to be translated into llvm ir")
                ir_return_type = self.type_map[function_def.returns.id]

        to_process = []
        if isinstance(function_def.body, typing.List):
            for statement in function_def.body:
                to_process.append(statement)

        node_list = paic_mlir.NodeList()
        while len(to_process) > 0:
            python_node = to_process[0]
            to_process.remove(python_node)
            if isinstance(python_node, _ast.Assign):
                if len(python_node.targets) != 1:
                    raise MLIRCompileError("tuple remaining in converted python")
                python_target = python_node.targets[0]
                if not isinstance(python_target, _ast.Name):
                    raise MLIRCompileError("target must be a name")
                result_name = python_target.id
                python_rhs = python_node.value
                expList = paic_mlir.ExpList()

                # TODO: support more than BinOp
                if isinstance(python_rhs, _ast.Call):
                    for a in python_rhs.args:
                        if isinstance(a, ast.Name):
                            if not symbols.__contains__(a.id):
                                raise MLIRCompileError("only local variables are referenceable")
                            ref = symbols[a.id]
                            node = paic_mlir.GetValNode(paic_mlir.Location(0, 0), ref.name(), ref.type())
                            expList.push_back(node)
                        elif isinstance(a, ast.Constant):
                            # TODO: support more than floats
                            node = paic_mlir.FloatNode(paic_mlir.Location(0, 0), a.value)
                            expList.push_back(node)
                    fnc_name = beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils.to_name(python_rhs.func)
                    if globals.__contains__(fnc_name):
                        value = globals[fnc_name]
                        if isinstance(value, typing.Callable):
                            alias = value.__str__()
                            if isinstance(value, beanmachine.ppl.compiler.runtime.builtin_function_or_method):
                                raise MLIRCompileError("Aliasing a built in method is not supported: " + alias)
                            lines, _ = inspect.getsourcelines(value)
                            source = "".join(beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils._unindent(lines))
                            module = ast.parse(source)
                            funcdef:_ast.FunctionDef = module.body[0]
                            globals[fnc_name] = funcdef.name
                        elif isinstance(value, str):
                            fnc_name = value

                elif isinstance(python_rhs, _ast.BinOp):
                    if not symbols.__contains__(python_rhs.left.id):
                        raise MLIRCompileError("only local variables are referencable")
                    if not symbols.__contains__(python_rhs.right.id):
                        raise MLIRCompileError("only local variables are referencable")

                    left = symbols[python_rhs.left.id]
                    right = symbols[python_rhs.right.id]
                    expList.push_back(paic_mlir.GetValNode(paic_mlir.Location(0, 0), left.name(), left.type()))
                    expList.push_back(paic_mlir.GetValNode(paic_mlir.Location(0, 0), right.name(), right.type()))

                    # TODO: nesting should be supported
                    if not isinstance(python_rhs.right, _ast.Name) or not isinstance(python_rhs.left, _ast.Name):
                        raise MLIRCompileError("no nesting allowed. Problem: " + python_target.id)
                    if isinstance(python_rhs.op, _ast.Mult):
                        # TODO: how to infer return type?
                        fnc_name = "times"
                    else:
                        raise MLIRCompileError("I only compile multiplication for now")
                call_node = paic_mlir.CallNode(paic_mlir.Location(0, 0), fnc_name, expList, self.floatType)
                var_node = paic_mlir.VarNode(paic_mlir.Location(0, 0), result_name, call_node.type(), call_node)
                node_list.push_back(var_node)
                symbols[result_name] = var_node
            elif isinstance(python_node, _ast.Return):
                if not isinstance(python_node.value, _ast.Name):
                    raise MLIRCompileError("no nesting allowed. Problem is return statement in " + function_def.name)
                if not symbols.__contains__(python_node.value.id):
                    raise MLIRCompileError("only local variables are referencable")
                ret_value = symbols[python_node.value.id]
                ret_node = paic_mlir.ReturnNode(paic_mlir.Location(0, 0),
                                                paic_mlir.GetValNode(paic_mlir.Location(0, 0), ret_value.name(),
                                                                     ret_value.type()))
                node_list.push_back(ret_node)

        body = paic_mlir.make_block_ptr(paic_mlir.Location(0, 0), node_list)
        python_function = paic_mlir.PythonFunction(paic_mlir.Location(0,0), function_def.name, ret_value.type(), param_list, body)
        return python_function