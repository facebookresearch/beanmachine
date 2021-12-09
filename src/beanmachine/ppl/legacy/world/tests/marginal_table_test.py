# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from beanmachine.ppl import random_variable as rv
from beanmachine.ppl.legacy.world.marginal_table import Table, Entry
from torch.distributions import Bernoulli


class MarginalTableTest(unittest.TestCase):
    def test_table_equality(self):
        @rv
        def a():
            return Bernoulli(0.5)

        @rv
        def b():
            return Bernoulli(a())

        table = Table()
        vals = torch.tensor([0.5, 0.5])
        e = Entry(a(), 2, {a()}, vals)
        table.add_entry(e)
        vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        e = Entry(b(), 2, {a(), b()}, vals)
        table.add_entry(e)

        table1 = Table()
        vals = torch.tensor([0.5, 0.5])
        e = Entry(a(), 2, {a()}, vals)
        table1.add_entry(e)
        self.assertNotEqual(table, table1)
        vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        e = Entry(b(), 2, {a(), b()}, vals)
        table1.add_entry(e)
        self.assertEqual(table, table1)
