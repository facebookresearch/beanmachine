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
import llvmlite.ir

from ctypes import CFUNCTYPE, c_double

import llvmlite.binding as llvm
from llvmlite.ir import Module


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
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    # Now add the module and make sure it is ready for execution
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


# assumes SSA form. Otherwise we won't have enough information to pop arguments
class GenerateLLVMIRVisitor:
    def create_IR(self, function_def: _ast.FunctionDef):
        stack_of_nodes = []
        to_process = []
        stack_of_nodes.append(function_def)
        # visit top down so that children are in data structure before parents. Note this only works if nested
        # expressions have been lifted
        while len(stack_of_nodes) > 0:
            parent_maybe = stack_of_nodes.pop()
            if isinstance(parent_maybe, _ast.FunctionDef):
                # parents are always puhsed before children
                to_process.append(parent_maybe)
                stack_of_nodes.append(parent_maybe.body)
                stack_of_nodes.append(parent_maybe.args)
            elif isinstance(parent_maybe, typing.List):
                # to mark a 'block' 🤔
                to_process.append(_ast.List())
                for s in parent_maybe.__reversed__():
                    stack_of_nodes.append(s)
            elif isinstance(parent_maybe, _ast.Name):
                to_process.append(parent_maybe)
            elif isinstance(parent_maybe, _ast.BinOp):
                to_process.append(parent_maybe)
                stack_of_nodes.append(parent_maybe.op)
                stack_of_nodes.append(parent_maybe.right)
                stack_of_nodes.append(parent_maybe.left)
            elif isinstance(parent_maybe, _ast.Mult):
                to_process.append(parent_maybe)
            elif isinstance(parent_maybe, _ast.Assign):
                to_process.append(parent_maybe)
                stack_of_nodes.append(parent_maybe.value)
                for t in parent_maybe.targets.__reversed__():
                    stack_of_nodes.append(t)
            elif isinstance(parent_maybe, _ast.arguments):
                for arg in parent_maybe.args:
                    stack_of_nodes.append(arg)
            elif isinstance(parent_maybe, _ast.arg):
                to_process.append(parent_maybe)
            elif isinstance(parent_maybe, _ast.Return):
                to_process.append(parent_maybe)
                stack_of_nodes.append(parent_maybe.value)

        substitutor = {}
        ir_nodes = []
        # iterate through the to_process nodes. When we encounter a Name that we have not encountered before, create


class llvmlite_test(unittest.TestCase):
    def get_function_ast(self, function: typing.Callable[[float], float]) -> _ast.FunctionDef:
        lines, starting_line = inspect.getsourcelines(foo)
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
        # mini test
        # All these initializations are required for code generation! (https://llvmlite.readthedocs.io/en/latest/user-guide/binding/examples.html)
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
        result = builder.fadd(a, b, name="res")
        builder.ret(result)
        # Print the module IR
        print(module)
        # execute the code!
        engine = create_execution_engine()
        # TODO: can we do this without turning it into a string first?
        mod = compile_ir_from_str(engine, module.__str__())
        # Look up the function pointer (a Python int)
        func_ptr = engine.get_function_address(func_name)

        # Run the function via ctypes
        cfunc = CFUNCTYPE(c_double, c_double, c_double)(func_ptr)
        res = cfunc(1.0, 3.5)
        print(func_name + "(...) =", res)

        ast = self.get_function_ast(foo)
        compiler = GenerateLLVMIRVisitor()
        llvm_ir = compiler.create_IR(ast)
        assert (True)
