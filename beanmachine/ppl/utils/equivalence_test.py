# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for partition functions from equivalence.py"""
import unittest
from typing import Any, Iterable

from beanmachine.ppl.utils.equivalence import partition_by_kernel, partition_by_relation


def _brace(s: str) -> str:
    return "{" + s + "}"


def _comma(s: Iterable[str]) -> str:
    return ",".join(s)


def _set_str(items: Iterable[Any]) -> str:
    return _brace(_comma(sorted({str(item) for item in items})))


def _set_set_str(results: Iterable[Any]) -> str:
    return _set_str([_set_str(eqv) for eqv in results])


class PartitionTest(unittest.TestCase):
    def test_partition_(self) -> None:
        """Tests for partition_kernel from equivalence.py"""

        def three_kernel(x: int) -> int:
            return (x % 3 + 3) % 3

        def three_relation(x: int, y: int) -> bool:
            return (x - y) % 3 == 0

        expected = """{{-1,-4,-7,2,5,8},{-2,-5,-8,1,4,7},{-3,-6,-9,0,3,6,9}}"""

        s = set(range(-9, 10))

        observed1 = _set_set_str(partition_by_relation(s, three_relation))
        observed2 = _set_set_str(partition_by_kernel(s, three_kernel))
        self.assertEqual(observed1, expected)
        self.assertEqual(observed2, expected)
