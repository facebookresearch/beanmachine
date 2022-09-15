# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch.distributions as dist
from beanmachine.ppl.legacy.world import Diff, Variable, World
from torch import tensor


class DiffStackTest(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), tensor(1.0))

    def test_diffstack_change(self):
        model = self.SampleModel()
        world = World()
        foo_key = model.foo()
        bar_key = model.bar()
        world.set_observations({bar_key: tensor(0.1)})
        diff_vars = world.diff_stack_
        foo_var = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.5),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.5)),
            children=set({bar_key}),
            transformed_value=tensor(0.5),
            jacobian=tensor(0.0),
        )
        diff_vars.add_node(foo_key, foo_var)
        bar_var = Variable(
            distribution=dist.Normal(tensor(0.5), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            transformed_value=tensor(0.1),
            jacobian=tensor(0.0),
        )
        diff_vars.add_node(bar_key, bar_var)
        self.assertEqual(diff_vars.get_node(bar_key), bar_var)
        self.assertEqual(diff_vars.get_node(foo_key), foo_var)
        self.assertListEqual(diff_vars.node_to_diffs_[foo_key], [0])
        self.assertListEqual(diff_vars.node_to_diffs_[bar_key], [0])

        diff_vars.add_diff(Diff())
        self.assertEqual(len(diff_vars.diff_stack_), 2)
        bar_var_copied = bar_var.copy()
        diff_vars.add_node(bar_key, bar_var_copied)
        self.assertListEqual(diff_vars.node_to_diffs_[bar_key], [0, 1])

        self.assertEqual(diff_vars.get_node(bar_key), bar_var_copied)
        self.assertEqual(diff_vars.get_node_earlier_version(bar_key), bar_var)
        self.assertEqual(bar_key in diff_vars.diff_var_, True)
        self.assertEqual(foo_key in diff_vars.diff_var_, True)
