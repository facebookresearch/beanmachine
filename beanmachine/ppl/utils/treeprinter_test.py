# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for print_tree from treeprinter.py"""
import unittest

from beanmachine.ppl.utils.treeprinter import print_tree


class TreePrinterTest(unittest.TestCase):
    def test_print_tree(self) -> None:
        """Tests for print_tree from treeprinter.py"""
        d = {"foo": 2, "bar": {"blah": [2, 3, {"abc": (6, 7, (5, 5, 6))}]}}
        observed = print_tree(d, unicode=False)
        expected = """dict
+-foo
| +-2
+-bar
  +-blah
    +-2
    +-3
    +-dict
      +-abc
        +-6
        +-7
        +-tuple
          +-5
          +-5
          +-6
"""
        self.assertEqual(observed, expected)
