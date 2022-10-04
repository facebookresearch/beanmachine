import ast
import inspect
import typing
import unittest

import paic2
import pytest
from beanmachine.paic2.inference.metaworld import MetaWorld
from beanmachine.paic2.inference.to_paic2_ast import paic2_ast_generator
from beanmachine.paic2.inference.utils import get_globals


def foo(a: float) -> float:
    return a


def fake_inference(world: MetaWorld):
    world.print()


def compile_to_mlir(py_func: typing.Callable):
    lines, _ = inspect.getsourcelines(py_func)
    source = "".join(lines)
    module = ast.parse(source)
    py_func_ast = module.body[0]
    mb = paic2.MLIRBuilder()
    to_paic = paic2_ast_generator()
    globals = get_globals(py_func)
    python_function = to_paic.python_ast_to_paic_ast(py_func_ast, globals)
    mb.print_func_name(python_function)


class BuildTest(unittest.TestCase):
    @pytest.mark.paic2
    def test_paic2_is_imported_math(self) -> None:
        compile_to_mlir(foo)

    @pytest.mark.paic2
    def test_paic2_is_imported_world(self) -> None:
        compile_to_mlir(fake_inference)


if __name__ == "__main__":
    unittest.main()
