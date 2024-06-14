import math

import beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_metal import to_metal

x = math.pow

@to_metal
def foo_builtin_alias(p1:float) -> float:
    i0 = p1 * p1
    i1 = x(i0, 2.0)
    return i1

@to_metal
def foo(p1:float) -> float:
    i0 = p1 * p1
    i1 = math.pow(i0, 2.0)
    return i1

def test_builtin_alias():
    try:
        call = foo_builtin_alias(3)
        assert False
    except beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast.MLIRCompileError as e:
        assert e.args[0] == "Aliasing a built in method is not supported: <built-in function pow>"

def test_power():
    try:
        call = foo(3.0)
        assert call == math.pow(3.0*3.0, 2.0)
    except beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast.MLIRCompileError as e:
        assert False

if __name__ == "__main__":
    test_power()
    test_builtin_alias()
