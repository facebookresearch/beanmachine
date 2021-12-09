# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for item_counter.py"""
import unittest

from beanmachine.ppl.utils.item_counter import ItemCounter


class ItemCounterTest(unittest.TestCase):
    def test_item_counter(self) -> None:
        i = ItemCounter()
        self.assertTrue("a" not in i.items)
        self.assertTrue("b" not in i.items)
        i.add_item("a")
        i.add_item("a")
        i.add_item("b")
        i.add_item("b")
        self.assertEqual(i.items["a"], 2)
        self.assertEqual(i.items["b"], 2)
        i.remove_item("b")
        i.remove_item("a")
        i.remove_item("a")
        self.assertTrue("a" not in i.items)
        self.assertEqual(i.items["b"], 1)
