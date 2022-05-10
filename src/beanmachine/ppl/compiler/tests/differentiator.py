import unittest
import beanmachine.ppl.compiler.differentiator

def foo(a: float) -> float:
    i0 = a * a
    i1 = i0 * i0
    return i1


class DifferentiatorTest(unittest.TestCase):
    def test_forward(self) -> None:
        value = 0.7
        # foo is a -> a*a*a*a so the forward should return 4*a^3
        derivative = beanmachine.ppl.compiler.differentiator.primal_and_derivative(foo, value)
        expectation = 4.0*value*value*value
        self.assertEqual(expectation, derivative)
