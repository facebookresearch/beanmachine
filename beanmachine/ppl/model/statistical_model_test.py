# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.model.statistical_model import (
    RVIdentifier,
    StatisticalModel,
)
from beanmachine.ppl.model.utils import Mode


class StatisticalModelTest(unittest.TestCase):
    def tearDown(self) -> None:
        # reset the StatisticalModel to prevent subsequent tests from failing
        StatisticalModel.reset()

    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

        @bm.random_variable
        def baz(self):
            return dist.Normal(self.foo(), self.bar())

    class SampleLargeModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

        @bm.random_variable
        def baz(self):
            return dist.Normal(self.foo(), self.bar())

        @bm.random_variable
        def foobar(self):
            return dist.Normal(self.baz(), self.bar())

        @bm.random_variable
        def bazbar(self, i):
            return dist.Normal(self.baz(), self.foo())

        @bm.random_variable
        def foobaz(self):
            return dist.Normal(self.bazbar(1), self.foo())

        @bm.functional
        def avg(self):
            return self.foo() + 1

    def test_rv_sample_assignment(self):
        model = self.SampleModel()
        world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        baz_key = model.baz()

        StatisticalModel.set_mode(Mode.INFERENCE)
        model.foo()
        model.bar()
        model.baz()

        foo_expected_parent = set()
        foo_expected_children = set({bar_key, baz_key})
        bar_expected_parent = set({foo_key})
        bar_expected_children = set({baz_key})
        baz_expected_parent = set({foo_key, bar_key})
        baz_expected_children = set()

        diff_vars = world.diff_.vars()
        self.assertEqual(foo_expected_children, diff_vars[foo_key].children)
        self.assertEqual(foo_expected_parent, diff_vars[foo_key].parent)
        self.assertEqual(bar_expected_children, diff_vars[bar_key].children)
        self.assertEqual(bar_expected_parent, diff_vars[bar_key].parent)
        self.assertEqual(baz_expected_children, diff_vars[baz_key].children)
        self.assertEqual(baz_expected_parent, diff_vars[baz_key].parent)

        foo_expected_dist = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        bar_expected_dist = dist.Normal(diff_vars[foo_key].value, torch.tensor(1.0))
        baz_expected_dist = dist.Normal(
            diff_vars[foo_key].value, diff_vars[bar_key].value
        )

        self.assertEqual(foo_expected_dist.mean, diff_vars[foo_key].distribution.mean)
        self.assertEqual(
            foo_expected_dist.stddev, diff_vars[foo_key].distribution.stddev
        )
        self.assertEqual(bar_expected_dist.mean, diff_vars[bar_key].distribution.mean)
        self.assertEqual(
            bar_expected_dist.stddev, diff_vars[bar_key].distribution.stddev
        )
        self.assertEqual(baz_expected_dist.mean, diff_vars[baz_key].distribution.mean)
        self.assertEqual(
            baz_expected_dist.stddev, diff_vars[baz_key].distribution.stddev
        )

    def test_rv_sample_assignment_with_large_model_with_index(self):
        model = self.SampleLargeModel()
        world = StatisticalModel.reset()
        foo_key = model.foo()
        bar_key = model.bar()
        baz_key = model.baz()
        foobar_key = model.foobar()
        bazbar_key = model.bazbar(1)
        foobaz_key = model.foobaz()
        query_key = model.avg()

        StatisticalModel.set_mode(Mode.INFERENCE)
        model.foo()
        model.bar()
        model.baz()
        model.foobar()
        model.bazbar(1)
        model.foobaz()

        foo_expected_parent = set()
        foo_expected_children = set({bar_key, baz_key, bazbar_key, foobaz_key})
        bar_expected_parent = set({foo_key})
        bar_expected_children = set({baz_key, foobar_key})
        baz_expected_parent = set({foo_key, bar_key})
        baz_expected_children = set({foobar_key, bazbar_key})
        foobar_expected_parent = set({baz_key, bar_key})
        foobar_expected_children = set()
        bazbar_expected_parent = set({baz_key, foo_key})
        bazbar_expected_children = set({foobaz_key})
        foobaz_expected_parent = set({bazbar_key, foo_key})
        foobaz_expected_children = set()

        diff_vars = world.diff_.vars()
        self.assertEqual(foo_expected_children, diff_vars[foo_key].children)
        self.assertEqual(foo_expected_parent, diff_vars[foo_key].parent)
        self.assertEqual(bar_expected_children, diff_vars[bar_key].children)
        self.assertEqual(bar_expected_parent, diff_vars[bar_key].parent)
        self.assertEqual(baz_expected_children, diff_vars[baz_key].children)
        self.assertEqual(baz_expected_parent, diff_vars[baz_key].parent)
        self.assertEqual(foobar_expected_children, diff_vars[foobar_key].children)
        self.assertEqual(foobar_expected_parent, diff_vars[foobar_key].parent)
        self.assertEqual(bazbar_expected_children, diff_vars[bazbar_key].children)
        self.assertEqual(bazbar_expected_parent, diff_vars[bazbar_key].parent)
        self.assertEqual(foobaz_expected_children, diff_vars[foobaz_key].children)
        self.assertEqual(foobaz_expected_parent, diff_vars[foobaz_key].parent)
        self.assertEqual(type(query_key), RVIdentifier)

        foo_expected_dist = dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        bar_expected_dist = dist.Normal(diff_vars[foo_key].value, torch.tensor(1.0))
        baz_expected_dist = dist.Normal(
            diff_vars[foo_key].value, diff_vars[bar_key].value
        )
        foobar_expected_dist = dist.Normal(
            diff_vars[baz_key].value, diff_vars[bar_key].value
        )
        bazbar_expected_dist = dist.Normal(
            diff_vars[baz_key].value, diff_vars[foo_key].value
        )
        foobaz_expected_dist = dist.Normal(
            diff_vars[bazbar_key].value, diff_vars[foo_key].value
        )

        self.assertEqual(foo_expected_dist.mean, diff_vars[foo_key].distribution.mean)
        self.assertEqual(
            foo_expected_dist.stddev, diff_vars[foo_key].distribution.stddev
        )
        self.assertEqual(bar_expected_dist.mean, diff_vars[bar_key].distribution.mean)
        self.assertEqual(
            bar_expected_dist.stddev, diff_vars[bar_key].distribution.stddev
        )
        self.assertEqual(baz_expected_dist.mean, diff_vars[baz_key].distribution.mean)
        self.assertEqual(
            baz_expected_dist.stddev, diff_vars[baz_key].distribution.stddev
        )
        self.assertEqual(
            foobar_expected_dist.mean, diff_vars[foobar_key].distribution.mean
        )
        self.assertEqual(
            foobar_expected_dist.stddev, diff_vars[foobar_key].distribution.stddev
        )
        self.assertEqual(
            bazbar_expected_dist.mean, diff_vars[bazbar_key].distribution.mean
        )
        self.assertEqual(
            bazbar_expected_dist.stddev, diff_vars[bazbar_key].distribution.stddev
        )
        self.assertEqual(
            foobaz_expected_dist.mean, diff_vars[foobaz_key].distribution.mean
        )
        self.assertEqual(
            foobaz_expected_dist.stddev, diff_vars[foobaz_key].distribution.stddev
        )
