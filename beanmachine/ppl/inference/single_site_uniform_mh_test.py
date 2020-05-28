# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)


class SingleSiteUniformMetropolisHastingsTest(unittest.TestCase):
    class SampleBernoulliModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Beta(torch.tensor(2.0), torch.tensor(2.0))

        @bm.random_variable
        def bar(self):
            return dist.Bernoulli(self.foo())

    class SampleCategoricalModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Dirichlet(torch.tensor([0.5, 0.5]))

        @bm.random_variable
        def bar(self):
            return dist.Categorical(self.foo())

    def test_single_site_uniform_mh_with_bernoulli(self):
        model = self.SampleBernoulliModel()
        mh = bm.SingleSiteUniformMetropolisHastings()
        foo_key = model.foo()
        bar_key = model.bar()
        mh.queries_ = [model.foo()]
        mh.observations_ = {model.bar(): torch.tensor(0.0)}
        mh._infer(10)
        self.assertEqual(isinstance(mh.proposer_, SingleSiteUniformProposer), True)
        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        world_vars = mh.world_.variables_.vars()
        self.assertEqual(foo_key in world_vars, True)
        self.assertEqual(bar_key in world_vars, True)
        self.assertEqual(foo_key in world_vars[bar_key].parent, True)
        self.assertEqual(bar_key in world_vars[foo_key].children, True)

    def test_single_site_uniform_mh_with_categorical(self):
        model = self.SampleCategoricalModel()
        mh = bm.SingleSiteUniformMetropolisHastings()
        foo_key = model.foo()
        bar_key = model.bar()
        mh.queries_ = [model.foo()]
        mh.observations_ = {model.bar(): torch.tensor(1.0)}
        mh._infer(10)

        self.assertEqual(isinstance(mh.proposer_, SingleSiteUniformProposer), True)
        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        world_vars = mh.world_.variables_.vars()
        self.assertEqual(foo_key in world_vars, True)
        self.assertEqual(bar_key in world_vars, True)
        self.assertEqual(foo_key in world_vars[bar_key].parent, True)
        self.assertEqual(bar_key in world_vars[foo_key].children, True)
