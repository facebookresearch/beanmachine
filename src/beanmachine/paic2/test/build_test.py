import ast
import inspect
import typing
import unittest

import paic2
import pytest


def foo():
    return 0


def compile_to_mlir(py_func: typing.Callable):
    lines, _ = inspect.getsourcelines(py_func)
    source = "".join(lines)
    module = ast.parse(source)
    py_func_ast = module.body[0]
    mb = paic2.MLIRBuilder()
    location = paic2.Location(py_func_ast.lineno, py_func_ast.col_offset)
    function = paic2.PythonFunction(location, str(py_func_ast.name), paic2.PrimitiveType(paic2.PrimitiveCode.Void), paic2.ParamList(), paic2.BlockNode(location, paic2.NodeList()))
    mb.print_func_name(function)


class BuildTest(unittest.TestCase):
    @pytest.mark.paic2
    def test_paic2_is_imported(self) -> None:
        compile_to_mlir(foo)


if __name__ == "__main__":
    compile_to_mlir(foo)
