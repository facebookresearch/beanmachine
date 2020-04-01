# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.model.statistical_model import StatisticalModel, sample
from beanmachine.ppl.model.utils import Mode
from beanmachine.ppl.world import Diff, Variable


class DiffStackTest(unittest.TestCase):
    class SampleModel(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), tensor(1.0))

    def test_diffstack_change(self):
        model = self.SampleModel()
        world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        StatisticalModel.set_mode(Mode.INFERENCE)
        world.set_observations({bar_key: tensor(0.1)})
        diff_vars = world.diff_stack_
        foo_var = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.5),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.5)),
            parent=set(),
            children=set({bar_key}),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=tensor(0.5),
            jacobian=tensor(0.0),
        )
        diff_vars.add_node(foo_key, foo_var)
        bar_var = Variable(
            distribution=dist.Normal(tensor(0.5), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            children=set(),
            proposal_distribution=None,
            extended_val=None,
            is_discrete=False,
            transforms=[],
            unconstrained_value=tensor(0.1),
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
