# what of instead of calling into C++ at runtime we built the machine code for inference and ran that?
# ideally, the inference algorithm is written in C++ and already compiled. The code that we would be compiling
# would be the constructed graph and a call to this compiled inference algorithm.
# further ideally the compiled compiled inference algorithm would be compiled from Python, not duplicated in C++
# The primary benefit of this approach is saving on development time. But that is a guess. What if compiling from Python to llvm is just as brutal?
# also, we can probably just compile the graph to start
from __future__ import print_function

import _ast
import ast
import inspect
import typing
import unittest

# ideally, foo would (1) build BMG graph (2) call the inference
# in order to call the inference, we need dependency symbols
import llvmlite
import llvmlite.binding
import llvmlite.ir

from ctypes import CFUNCTYPE, c_double, c_float

import llvmlite.binding as llvm
from llvmlite.binding import ModuleRef, ExecutionEngine
from llvmlite.ir import Module, Instruction


class LLVMCompileError(Exception):
    def __init__(self, message: str):
        self.__cause__ = message


def foo(a: float) -> float:
    i0 = a * a
    return i0


def create_execution_engine():
    """
    Create an ExecutionEngine suitable for JIT code generation on
    the host CPU.  The engine is reusable for an arbitrary number of
    modules.
    """
    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    # And an execution engine with an empty backing module
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir_from_str(engine, llvm_ir: str):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod: ModuleRef = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


def compile_ir(engine: ExecutionEngine, mod: ModuleRef):
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


# assumes SSA form. Otherwise we won't have enough information to pop arguments
class GenerateLLVMIRVisitor:
    def __init__(self):
        self.type_map = {'float': llvmlite.ir.FloatType(), 'double': llvmlite.ir.DoubleType}

    def llvm_function(self, function_def: _ast.FunctionDef, module: llvm.ir.Module) -> llvmlite.ir.Function:
        ir_param_types = []
        substitutor: typing.Dict[str, llvmlite.ir.Argument] = {}
        # validate function
        if isinstance(function_def, _ast.FunctionDef):
            for a in function_def.args.args:
                if isinstance(a, _ast.arg):
                    if isinstance(a.annotation, _ast.Name):
                        if not self.type_map.__contains__(a.annotation.id):
                            raise LLVMCompileError("all arguments must have types in order to be translated into llvm ir")
                        ir_param_types.append(self.type_map[a.annotation.id])

            if isinstance(function_def.returns, _ast.Name):
                return_type = function_def.returns.id
                if not self.type_map.__contains__(return_type):
                    raise LLVMCompileError("return types must have types in order to be translated into llvm ir")
                ir_return_type = self.type_map[function_def.returns.id]

        to_process = []
        # initialize
        function_signature = llvmlite.ir.FunctionType(ir_return_type, ir_param_types)
        func = llvmlite.ir.Function(module, function_signature, name=function_def.name)
        i = 0
        for arg in func.args:
            python_arg = function_def.args.args[i]
            i = i+1
            substitutor[python_arg.arg] = arg

        block = func.append_basic_block(name="entry")
        builder = llvmlite.ir.IRBuilder(block)

        if isinstance(function_def.body, typing.List):
            for statement in function_def.body:
                to_process.append(statement)

        while len(to_process) > 0:
            python_node = to_process[0]
            to_process.remove(python_node)
            if isinstance(python_node, _ast.Assign):
                if len(python_node.targets) != 1:
                    raise LLVMCompileError("tuple remaining in converted python")
                python_target = python_node.targets[0]
                if not isinstance(python_target, _ast.Name):
                    raise LLVMCompileError("target must be a name")
                result_name = python_target.id
                # TODO: support more than mult
                python_rhs = python_node.value
                if not isinstance(python_rhs, _ast.BinOp):
                    raise LLVMCompileError("Prototype only supports binary operations")
                op = python_rhs.op
                left = substitutor[python_rhs.left.id]
                right = substitutor[python_rhs.right.id]
                if not isinstance(python_rhs.right, _ast.Name) or not isinstance(python_rhs.left, _ast.Name):
                    raise LLVMCompileError("no nesting allowed. Problem: " + python_target.id)
                if isinstance(op, _ast.Mult):
                    if isinstance(left.type, llvm.ir.FloatType) and isinstance(right.type, llvm.ir.FloatType):
                        instruction = builder.fmul(left, right, result_name)
                    else:
                        instruction = builder.mul(left, right, result_name)
                else:
                    raise LLVMCompileError("I only compile multiplication for now")
                substitutor[result_name] = instruction
            elif isinstance(python_node, _ast.Return):
                if not isinstance(python_node.value, _ast.Name):
                    raise LLVMCompileError("no nesting allowed. Problem is return statement in " + function_def.name)
                instruction = substitutor[python_node.value.id]
                builder.ret(instruction)
        return func


class llvmlite_test(unittest.TestCase):
    def get_function_ast(self, function: typing.Callable[[float], float]) -> _ast.FunctionDef:
        lines, starting_line = inspect.getsourcelines(function)
        source = "".join(lines)
        module = ast.parse(source, mode='exec')
        stmts: [_ast.AST] = module.body
        if len(stmts) == 1:
            function_def_maybe = stmts[0]
            if isinstance(function_def_maybe, _ast.FunctionDef):
                function_def: _ast.FunctionDef = function_def_maybe
                return function_def
        raise

    def test_compilation(self) -> None:
        # mini test All these initializations are required for code generation! (
        # https://llvmlite.readthedocs.io/en/latest/user-guide/binding/examples.html)
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        # Create some useful types

        double = llvmlite.ir.DoubleType()
        fnty = llvmlite.ir.FunctionType(double, (double, double))
                # Create an empty module...
        module: Module = llvmlite.ir.Module(name=__file__)
        # and declare a function named "fpadd" inside it
        func_name = "fpadd"
        func = llvmlite.ir.Function(module, fnty, name=func_name)

        # Now implement the function
        block = func.append_basic_block(name="entry")
        builder = llvmlite.ir.IRBuilder(block)
        a, b = func.args
        result: Instruction = builder.fadd(a, b, name="res")
        builder.ret(result)
        # # Print the module IR
        print(module)
        # # execute the code!
        # engine: ExecutionEngine = create_execution_engine()
        # # TODO: can we do this without turning it into a string first?
        # mod = compile_ir_from_str(engine, module.__str__())
        # # Look up the function pointer (a Python int)
        # func_ptr = engine.get_function_address(func_name)
        #
        # # Run the function via ctypes
        # cfunc = CFUNCTYPE(c_double, c_double, c_double)(func_ptr)
        # res = cfunc(1.0, 3.5)
        # print(func_name + "(...) =", res)


        compiler = GenerateLLVMIRVisitor()
        module = llvmlite.ir.Module(name=__file__)
        ast = self.get_function_ast(foo)
        llvm_ir = compiler.llvm_function(ast, module)
        print(module)
        engine: ExecutionEngine = create_execution_engine()
        # # TODO: can we do this without turning it into a string first?
        mod = compile_ir_from_str(engine, module.__str__())
        # Look up the function pointer (a Python int)
        func_ptr = engine.get_function_address(llvm_ir.name)

        # Run the function via ctypes
        cfunc = CFUNCTYPE(c_float, c_float)(func_ptr)
        x = 3.5
        res = cfunc(x)
        print(llvm_ir.name + "(...) =", res)
        assert (res == foo(x))
