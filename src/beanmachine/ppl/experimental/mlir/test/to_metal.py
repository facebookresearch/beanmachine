import inspect
import ast

from paic_mlir import MLIRBuilder

from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.to_paic_ast import paic_ast_generator
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils import _unindent
from typing import Callable
from beanmachine.ppl.compiler.paic.mlir.test_paic_mlir.utils import _get_globals

mb = MLIRBuilder()

# takes a callable and returns an ast
def to_metal(callable:Callable):
    # should return callable
    def wrapper(*args, **kwargs):
        lines, _ = inspect.getsourcelines(callable)
        source = "".join(_unindent(lines))
        module = ast.parse(source)
        funcdef = module.body[0]
        # TODO: collect ASTs of query methods
        to_paic = paic_ast_generator()
        globals = _get_globals(callable)
        python_function = to_paic.python_ast_to_paic_ast(funcdef, globals)
        # TODO: pass the paic ast to the import function instead
        arg = float(args[0])
        result = mb.to_metal(python_function, arg)
        return result
    return wrapper