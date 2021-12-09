# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for memoize.py"""
import unittest

from beanmachine.ppl.utils.memoize import memoize


count1 = 0


def fib(n):
    global count1
    count1 = count1 + 1
    return 1 if n <= 1 else fib(n - 1) + fib(n - 2)


count2 = 0


@memoize
def fib_mem(n):
    global count2
    count2 = count2 + 1
    return 1 if n <= 1 else fib_mem(n - 1) + fib_mem(n - 2)


class MemoizeTest(unittest.TestCase):
    """Tests for memoize.py"""

    def test_memoize(self) -> None:
        """Tests for memoize.py"""
        global count1
        global count2
        f10 = fib(10)
        self.assertEqual(f10, 89)
        self.assertEqual(count1, 177)
        f10 = fib_mem(10)
        self.assertEqual(f10, 89)
        self.assertEqual(count2, 11)
