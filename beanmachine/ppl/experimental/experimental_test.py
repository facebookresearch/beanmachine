# Copyright (c) Facebook, Inc. and its affiliates.
import types
import unittest

import beanmachine.ppl as bm


class ExperimentalTest(unittest.TestCase):
    def test_can_import(self):
        self.assertTrue(isinstance(bm.experimental, types.ModuleType))
