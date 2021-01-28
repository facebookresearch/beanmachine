# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for fold_constants.py"""
import ast
import unittest

import astor
from beanmachine.ppl.utils.fold_constants import (
    _fix_associative_ops,
    _move_constants,
    fold,
)
from beanmachine.ppl.utils.rules import TryMany as many


class ConstantFoldTest(unittest.TestCase):
    def disabled_test_constant_fold(self) -> None:
        """Tests for fold_constants.py"""

        # PYTHON VERSIONING ISSUE
        # TODO: There is some difference in the parse trees in the new version of
        # Python that we are not expecting. Until we understand what is going on,
        # disable this test.

        self.maxDiff = None

        source = """
~0; -1; +2; 3+4*5-6**2; 5&6; 7|8; 9^10
11/0; 12/4; 14%0; 15%5; 4>>1; 4<<1"""
        m = ast.parse(source)
        result = fold(m)
        expected = """
-1; -1; 2; -13; 4; 15; 3
11/0; 3.0; 14%0; 0; 2; 8"""
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))

        source = """
False or False or False or False
False or True or False or False
True and True and True and True
True and False and False and True
x = 1 if not ((True or False) and True) else 2
"""
        m = ast.parse(source)
        result = fold(m)
        expected = "False; True; True; False; x = 2"
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(expected)))

        # Python has an unusual design of its comparison operators; they are
        # n-ary, and x op1 y op2 z  has the semantics "(x op1 y) and (y op2 z)".
        #
        # I've added a rule to the constant folder that detects comparisons
        # where *all* the operands are constant numbers and lowers them first
        # into the binary operator form, and lowers the binary operator form
        # to True or False. In the example below, we first lower to
        # "1 < 2 and 2 < 3", then recurse to produce "True and True", and then
        # we run the constant folding rule on the node again to produce True.
        # This gets us to a fixpoint.

        source = """1 < 2 < 3"""
        m = ast.parse(source)
        result = fold(m)
        expected = "True"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_constant_fold_2(self) -> None:
        """Tests for fold_constants.py"""
        source = """a + (b + c * (d * (e + (f + g))))"""
        m = ast.parse(source)
        fao = many(_fix_associative_ops)
        result = fao(m).expect_success()
        expected = "a + b + c * d * (e + f + g)"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        source = """(a - (b - c)) + (d - (e + f)) + (g + (h - i))"""
        m = ast.parse(source)
        result = fao(m).expect_success()
        expected = "a - b + c + d - e - f + g + h - i"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        source = """(a / (b / c)) / (d / (e * f)) * (g * (h / i))"""
        m = ast.parse(source)
        result = fao(m).expect_success()
        expected = "a / b * c / d * e * f * g * h / i"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        # We should be able to fold constants in expressions like this by
        # first fixing the associative operators...
        source = """((a - 2) + (b - 3)) + ((c + 4 + 5) - (6 + d - 7))"""
        m = ast.parse(source)
        assoc_fixed = fao(m).expect_success()
        expected = "a - 2 + b - 3 + c + 4 + 5 - 6 - d + 7"
        self.assertEqual(astor.to_source(assoc_fixed).strip(), expected.strip())

        # ... and then creating a fixpoint combinator on _move_constants:
        result = many(_move_constants)(assoc_fixed).expect_success()
        expected = "a + 5 + b + c - d"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_constant_fold_3(self) -> None:
        """Tests for fold_constants.py"""
        source = """x + (1 if (1 < 2 < (3 + 4)) else 2) * (3 * y * 4) + 5"""
        m = ast.parse(source)
        result = fold(m)
        expected = "x + 5 + 12 * y"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_constant_fold_4(self) -> None:
        """Tests for fold_constants.py"""
        source = """-tensor(1) + x + tensor(2) + 3 + tensor(4) * y * tensor(5)"""
        m = ast.parse(source)
        result = fold(m)
        expected = "torch.tensor(4) + x + torch.tensor(20) * y"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        source = """-tensor([1, 2]) + x + tensor([4, 8])"""
        m = ast.parse(source)
        result = fold(m)
        expected = "torch.tensor([3, 6]) + x"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_constant_fold_5(self) -> None:
        """Tests for fold_constants.py"""
        # These tests produce errors, but folding should still succeed and
        # reach a fixpoint.
        source = """tensor([1, 2, 3]) + x + tensor([1, 2])"""
        m = ast.parse(source)
        result = fold(m)
        expected = "tensor([1, 2, 3]) + tensor([1, 2]) + x"
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

    def test_constant_fold_6(self) -> None:
        """Tests for fold_constants.py"""
        # We can fold certain pure functions, like math.log or torch.log.
        source = (
            "torch.log(tensor([1.0, 2.0, 3.0])) + x + "
            + "tensor([math.log(2.7), math.acos(0.1), 3.0]) + "
            + "torch.exp(torch.neg(tensor([-1.0, -2.0, -3.0]))) + "
            + "tensor([math.exp(1.0), 2.0, 3.0])"
        )
        m = ast.parse(source)
        result = fold(m)
        expected = (
            "torch.tensor([6.429815292358398, 11.552831649780273, "
            + "27.18414878845215]) + x"
        )
        self.assertEqual(astor.to_source(result).strip(), expected.strip())

        # If the code is wrong and throws an exception, folding is a no-op.
        # Note that we do manage to fold log of -1.0 down to NaN.  However
        # we do not consider "float('nan')" to be a constant float, so
        # that expression will not participate in folding afterwards.
        source = """torch.log(tensor([-1.0])) + x + tensor([math.log(0.0), 2.0, 3.0])"""
        m = ast.parse(source)
        result = fold(m)
        expected = (
            "torch.tensor([float('nan')]) + x + tensor([math.log(0.0), 2.0, 3.0])"
        )
        self.assertEqual(astor.to_source(result).strip(), expected.strip())
