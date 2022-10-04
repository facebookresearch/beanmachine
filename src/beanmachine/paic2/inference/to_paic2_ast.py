import _ast
import ast
import inspect
import typing

import paic2

from beanmachine.paic2.inference.utils import to_name
from beanmachine.ppl.compiler.runtime import builtin_function_or_method


class MLIRCompileError(Exception):
    pass


class paic2_ast_generator:
    def __init__(self, world_length: int = 0):
        self.unitType = paic2.PrimitiveType(paic2.PrimitiveCode.Void)
        self.floatType = paic2.PrimitiveType(paic2.PrimitiveCode.Float)
        self.worldType = paic2.WorldType(paic2.PrimitiveCode.Double, world_length)
        self.type_map = {"float": self.floatType, "MetaWorld": self.worldType}

    def _params(self, function_def: _ast.FunctionDef) -> paic2.ParamList():
        param_list = paic2.ParamList()
        return_type = self.unitType
        if isinstance(function_def, _ast.FunctionDef):
            for a in function_def.args.args:
                if isinstance(a, _ast.arg):
                    if isinstance(a.annotation, _ast.Name):
                        if not self.type_map.__contains__(a.annotation.id):
                            raise MLIRCompileError(
                                "all arguments must have types in order to be translated into mlir ir"
                            )
                        param_node = paic2.ParamNode(
                            paic2.Location(0, 0), a.arg, self.type_map[a.annotation.id]
                        )
                        param_list.push_back(param_node)
            if isinstance(function_def.returns, _ast.Name):
                return_type = function_def.returns.id
                if not self.type_map.__contains__(return_type):
                    raise MLIRCompileError(
                        "return types must have types in order to be translated into llvm ir"
                    )
        return (param_list, return_type)

    def _process_binary_op_node(
        self, binary_op: ast.BinOp, symbols: typing.Dict[str, paic2.DeclareValNode]
    ) -> paic2.CallNode:
        expList = paic2.ExpList()
        if not symbols.__contains__(binary_op.left.id):
            raise MLIRCompileError("only local variables are referencable")
        if not symbols.__contains__(binary_op.right.id):
            raise MLIRCompileError("only local variables are referencable")

        left = symbols[binary_op.left.id]
        right = symbols[binary_op.right.id]
        expList.push_back(
            paic2.GetValNode(paic2.Location(0, 0), left.name(), left.type())
        )
        expList.push_back(
            paic2.GetValNode(paic2.Location(0, 0), right.name(), right.type())
        )

        # TODO: nesting should be supported
        if not isinstance(binary_op.right, _ast.Name) or not isinstance(
            binary_op.left, _ast.Name
        ):
            raise MLIRCompileError("no nesting allowed. Problem: " + binary_op.id)
        if isinstance(binary_op.op, _ast.Mult):
            # TODO: how to infer return type?
            fnc_name = "times"
        else:
            raise MLIRCompileError("I only compile multiplication for now")
        return paic2.CallNode(paic2.Location(0, 0), fnc_name, expList, self.floatType)

    def _process_call_node(
        self,
        call_node: _ast.Call,
        symbols: typing.Dict[str, paic2.DeclareValNode],
        globals: typing.Dict,
    ) -> paic2.CallNode:
        expList = paic2.ExpList()
        for a in call_node.args:
            if isinstance(a, ast.Name):
                if not symbols.__contains__(a.id):
                    raise MLIRCompileError("only local variables are referenceable")
                ref = symbols[a.id]
                node = paic2.GetValNode(paic2.Location(0, 0), ref.name(), ref.type())
                expList.push_back(node)
            elif isinstance(a, ast.Constant):
                # TODO: support more than floats
                node = paic2.FloatNode(paic2.Location(0, 0), a.value)
                expList.push_back(node)
        fnc_name = to_name(call_node.func)
        if globals.__contains__(fnc_name):
            value = globals[fnc_name]
            if isinstance(value, typing.Callable):
                alias = value.__str__()
                if isinstance(value, builtin_function_or_method):
                    raise MLIRCompileError(
                        "Aliasing a built in method is not supported: " + alias
                    )
                lines, _ = inspect.getsourcelines(value)
                source = "".join(lines)
                module = ast.parse(source)
                funcdef: _ast.FunctionDef = module.body[0]
                globals[fnc_name] = funcdef.name
            elif isinstance(value, str):
                fnc_name = value
        return paic2.CallNode(paic2.Location(0, 0), fnc_name, expList, self.floatType)

    def _process_expression_node(
        self, python_node: _ast.Expr, symbols: typing.Dict[str, paic2.DeclareValNode]
    ) -> paic2.CallNode:
        node = python_node.value
        if isinstance(node, ast.Call):
            expList = paic2.ExpList()
            fnc = node.func
            fnc_name = ""
            receiver = None
            if isinstance(fnc, ast.Attribute):
                if isinstance(fnc.value, ast.Name):
                    # is the expression a call on a receiver?
                    if symbols.__contains__(fnc.value.id):
                        ref = symbols[fnc.value.id]
                        fnode = paic2.GetValNode(
                            paic2.Location(0, 0), ref.name(), ref.type()
                        )
                        fnc_name = fnc.attr
                        receiver = fnode
            for a in node.args:
                if isinstance(a, ast.Name):
                    if not symbols.__contains__(a.id):
                        raise MLIRCompileError("only local variables are referenceable")
                    ref = symbols[a.id]
                    fnode = paic2.GetValNode(
                        paic2.Location(0, 0), ref.name(), ref.type()
                    )
                    expList.push_back(fnode)
                elif isinstance(a, ast.Constant):
                    # TODO: support more than floats
                    fnode = paic2.FloatNode(paic2.Location(0, 0), a.value)
                    expList.push_back(fnode)
            if fnc_name == "":
                fnc_name = to_name(node.func)
            if receiver is None:
                call_node = paic2.CallNode(
                    paic2.Location(0, 0), fnc_name, expList, self.unitType
                )
            else:
                call_node = paic2.CallNode(
                    paic2.Location(0, 0), fnc_name, expList, receiver, self.unitType
                )
            return call_node
        else:
            raise MLIRCompileError(
                "only call nodes are supported as statement expressions"
            )

    def _process_return_node(
        self,
        python_node: _ast.Return,
        function_def: _ast.FunctionDef,
        symbols: typing.Dict[str, paic2.DeclareValNode],
    ) -> paic2.ReturnNode:
        if not isinstance(python_node.value, _ast.Name):
            raise MLIRCompileError(
                "no nesting allowed. Problem is return statement in "
                + function_def.name
            )
        if not symbols.__contains__(python_node.value.id):
            raise MLIRCompileError("only local variables are referencable")
        ret_value = symbols[python_node.value.id]
        ret_node = paic2.ReturnNode(
            paic2.Location(0, 0),
            paic2.GetValNode(paic2.Location(0, 0), ret_value.name(), ret_value.type()),
        )
        return ret_node

    def python_ast_to_paic_ast(
        self, function_def: _ast.FunctionDef, globals: typing.Dict
    ) -> paic2.PythonFunction:
        symbols: typing.Dict[str, paic2.DeclareValNode] = {}
        ret_value_type = self.unitType

        param_list, return_type = self._params(function_def)
        for param in param_list:
            symbols[param.name()] = param

        to_process = []
        if isinstance(function_def.body, typing.List):
            for statement in function_def.body:
                to_process.append(statement)

        node_list = paic2.NodeList()
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

                # TODO: support more than BinOp
                if isinstance(python_rhs, _ast.Call):
                    call_node = self._process_call_node(python_rhs, symbols, globals)
                elif isinstance(python_rhs, _ast.BinOp):
                    call_node = self._process_binary_op_node(python_rhs, symbols)
                else:
                    raise MLIRCompileError(
                        "only binary op and call nodes are supported as rhs of variables for now"
                    )

                var_node = paic2.VarNode(
                    paic2.Location(0, 0), result_name, call_node.type(), call_node
                )
                node_list.push_back(var_node)
                symbols[result_name] = var_node
            elif isinstance(python_node, _ast.Expr):
                call_node = self._process_expression_node(python_node, symbols)
                node_list.push_back(call_node)
            elif isinstance(python_node, _ast.Return):
                ret_node = self._process_return_node(python_node, function_def, symbols)
                ret_value_type = ret_node.value().type()
                node_list.push_back(ret_node)
            else:
                raise MLIRCompileError("Unsupported statement: " + str(python_node))

        python_function = paic2.PythonFunction(
            paic2.Location(0, 0),
            function_def.name,
            ret_value_type,
            param_list,
            paic2.BlockNode(paic2.Location(0, 0), node_list),
        )
        return python_function
