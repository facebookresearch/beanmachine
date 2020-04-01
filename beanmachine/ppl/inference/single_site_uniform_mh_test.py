# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.proposer.single_site_uniform_proposer import (
    SingleSiteUniformProposer,
)
from beanmachine.ppl.inference.single_site_uniform_mh import (
    SingleSiteUniformMetropolisHastings,
)
from beanmachine.ppl.model.statistical_model import sample


class SingleSiteUniformMetropolisHastingsTest(unittest.TestCase):
    class SampleBernoulliModel(object):
        @sample
        def foo(self):
            return dist.Beta(torch.tensor(2.0), torch.tensor(2.0))

        @sample
        def bar(self):
            return dist.Bernoulli(self.foo())

    class SampleCategoricalModel(object):
        @sample
        def foo(self):
            return dist.Dirichlet(torch.tensor([0.5, 0.5]))

        @sample
        def bar(self):
            return dist.Categorical(self.foo())

    def test_single_site_uniform_mh_with_bernoulli(self):
        model = self.SampleBernoulliModel()
        mh = SingleSiteUniformMetropolisHastings()
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
        mh = SingleSiteUniformMetropolisHastings()
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
