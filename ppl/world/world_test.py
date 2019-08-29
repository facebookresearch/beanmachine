# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch.distributions as dist
import torch.tensor as tensor
from beanmachine.ppl.model.statistical_model import StatisticalModel, sample
from beanmachine.ppl.model.utils import Mode
from beanmachine.ppl.world.variable import Variable


class WorldTest(unittest.TestCase):
    class SampleModel(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), tensor(1.0))

    class SampleModelWithParentUpdate(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def baz(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def bar(self):
            if self.foo().item() > 0.3:
                return dist.Normal(self.foo(), tensor(1.0))
            return dist.Normal(self.baz(), tensor(1.0))

    class SampleLargeModelUpdate(object):
        @sample
        def foo(self):
            return dist.Normal(tensor(0.0), tensor(1.0))

        @sample
        def baz(self):
            return dist.Normal(self.foo(), tensor(1.0))

        @sample
        def foobar(self):
            return dist.Normal(self.foo(), tensor(1.0))

        @sample
        def bar(self):
            if self.foo().item() < 0.3:
                return dist.Normal(self.foo(), tensor(1.0))
            if self.foo().item() < 0.5:
                return dist.Normal(self.baz(), tensor(1.0))
            return dist.Normal(self.foobar(), tensor(1.0))

    def test_world_change(self):
        model = self.SampleModel()
        stack, world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        StatisticalModel.set_mode(Mode.INFERENCE)
        world.variables_[foo_key] = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.5),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.5)),
            parent=set(),
            children=set({bar_key}),
        )

        world.variables_[bar_key] = Variable(
            distribution=dist.Normal(tensor(0.5), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            children=set(),
        )

        children_log_update, world_log_update, node_log_update = world.propose_change(
            foo_key, tensor(0.25), stack
        )

        expected_children_log_update = dist.Normal(tensor(0.25), tensor(1.0)).log_prob(
            tensor(0.1)
        ) - dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1))
        expected_world_log_update = expected_children_log_update + node_log_update

        self.assertAlmostEqual(
            children_log_update.item(), expected_children_log_update.item(), places=3
        )
        self.assertAlmostEqual(
            world_log_update.item(), expected_world_log_update.item(), places=3
        )

    def test_world_change_with_parent_update_and_new_node(self):
        model = self.SampleModelWithParentUpdate()
        stack, world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        baz_key = model.baz()
        StatisticalModel.set_mode(Mode.INFERENCE)

        world.variables_[foo_key] = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.5),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.5)),
            parent=set(),
            children=set({bar_key}),
        )

        world.variables_[bar_key] = Variable(
            distribution=dist.Normal(tensor(0.5), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            children=set(),
        )

        children_log_update, world_log_update, _ = world.propose_change(
            foo_key, tensor(0.25), stack
        )

        expected_children_log_update = (
            dist.Normal(world.diff_[baz_key].value, tensor(1.0))
            .log_prob(tensor(0.1))
            .sub(dist.Normal(tensor(0.5), tensor(1.0)).log_prob(tensor(0.1)))
        )

        expected_node_update = 0.0938
        expected_world_update = expected_children_log_update.add(
            tensor(expected_node_update)
        ).add(
            dist.Normal(tensor(0.0), tensor(1.0)).log_prob(world.diff_[baz_key].value)
        )
        self.assertAlmostEqual(
            children_log_update.item(), expected_children_log_update.item(), places=3
        )
        self.assertAlmostEqual(
            world_log_update.item(), expected_world_update.item(), places=3
        )
        self.assertEqual((baz_key) in world.diff_[bar_key].parent, True)

    def test_world_change_with_multiple_parent_update(self):
        model = self.SampleLargeModelUpdate()
        stack, world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        baz_key = model.baz()
        foobar_key = model.foobar()
        StatisticalModel.set_mode(Mode.INFERENCE)

        world.variables_[foo_key] = Variable(
            distribution=dist.Normal(tensor(0.0), tensor(1.0)),
            value=tensor(0.2),
            log_prob=dist.Normal(tensor(0.0), tensor(1.0)).log_prob(tensor(0.2)),
            parent=set(),
            children=set({bar_key}),
        )

        world.variables_[bar_key] = Variable(
            distribution=dist.Normal(tensor(0.2), tensor(1.0)),
            value=tensor(0.1),
            log_prob=dist.Normal(tensor(0.2), tensor(1.0)).log_prob(tensor(0.1)),
            parent=set({foo_key}),
            children=set(),
        )

        expected_node_update = (
            dist.Normal(tensor(0.0), tensor(1.0))
            .log_prob(0.35)
            .sub(dist.Normal(tensor(0.0), tensor(1.0)).log_prob(0.2))
        )

        children_log_update, world_log_update, _ = world.propose_change(
            foo_key, tensor(0.35), stack
        )

        expected_children_log_update = (
            dist.Normal(world.diff_[baz_key].value, tensor(1.0))
            .log_prob(tensor(0.1))
            .sub(dist.Normal(tensor(0.2), tensor(1.0)).log_prob(tensor(0.1)))
        )

        expected_world_update = expected_children_log_update.add(
            expected_node_update
        ).add(
            dist.Normal(world.diff_[foo_key].value, tensor(1.0)).log_prob(
                world.diff_[baz_key].value
            )
        )

        self.assertAlmostEqual(
            children_log_update.item(), expected_children_log_update.item(), places=3
        )
        self.assertAlmostEqual(
            world_log_update.item(), expected_world_update.item(), places=3
        )
        self.assertEqual((baz_key) in world.diff_[bar_key].parent, True)

        world.accept_diff()

        expected_node_update = (
            dist.Normal(tensor(0.0), tensor(1.0))
            .log_prob(0.55)
            .sub(dist.Normal(tensor(0.0), tensor(1.0)).log_prob(0.35))
        )

        children_log_update, world_log_update, _ = world.propose_change(
            foo_key, tensor(0.55), stack
        )

        expected_children_log_update = (
            dist.Normal(world.diff_[foobar_key].value, tensor(1.0))
            .log_prob(tensor(0.1))
            .sub(
                dist.Normal(world.diff_[baz_key].value, tensor(1.0)).log_prob(
                    tensor(0.1)
                )
            )
            .add(
                dist.Normal(tensor(0.55), tensor(1.0))
                .log_prob(world.diff_[baz_key].value)
                .sub(
                    dist.Normal(tensor(0.35), tensor(1.0)).log_prob(
                        world.diff_[baz_key].value
                    )
                )
            )
        )

        expected_world_update = expected_children_log_update.add(
            tensor(expected_node_update)
        ).add(
            dist.Normal(world.diff_[foo_key].value, tensor(1.0)).log_prob(
                world.diff_[foobar_key].value
            )
        )

        self.assertAlmostEqual(
            children_log_update.item(), expected_children_log_update.item(), places=3
        )
        self.assertAlmostEqual(
            world_log_update.item(), expected_world_update.item(), places=3
        )
        self.assertEqual((foobar_key) in world.diff_[bar_key].parent, True)
