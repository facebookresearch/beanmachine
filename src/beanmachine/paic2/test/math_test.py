import math
import unittest

import pytest

from beanmachine.paic2.inference.paic2_decorators import to_hardware
from beanmachine.paic2.inference.to_paic2_ast import MLIRCompileError

x = math.pow


@to_hardware
def foo_builtin_alias(p1: float) -> float:
    i0 = p1 * p1
    i1 = x(i0, 2.0)
    return i1


@to_hardware
def foo(p1: float) -> float:
    i0 = p1 * p1
    i1 = math.pow(i0, 2.0)
    return i1


class MathTest(unittest.TestCase):
    @pytest.mark.paic2
    def test_builtin_alias(self):
        try:
            foo_builtin_alias(3)
            self.assertFalse(True)
        except MLIRCompileError as e:
            self.assertTrue(
                e.args[0]
                == "Aliasing a built in method is not supported: <built-in function pow>"
            )

    @pytest.mark.paic2
    def test_power(self):
        try:
            call = foo(3.0)
            assert call == math.pow(3.0 * 3.0, 2.0)
        except MLIRCompileError:
            assert False


if __name__ == "__main__":
    unittest.main()
