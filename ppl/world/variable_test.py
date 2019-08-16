# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from beanmachine.ppl.world.variable import Variable


class TestVariable(unittest.TestCase):
    def test_variable_types(self):
        with self.assertRaises(ValueError):
            Variable(1, 1, 1, 1, 1)
