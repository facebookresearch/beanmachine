# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.model.statistical_model import sample


class SingleSiteAncestralMetropolisHastingsTest(unittest.TestCase):
    class SampleModel(object):
        @sample
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_single_site_ancestral_mh(self):
        model = self.SampleModel()
        mh = SingleSiteAncestralMetropolisHastings(
            [model.foo()], {model.bar(): torch.tensor(0.0)}
        )
        foo_key = model.foo()
        bar_key = model.bar()
        mh._infer(10)

        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        self.assertEqual(foo_key in mh.world_.variables_, True)
        self.assertEqual(bar_key in mh.world_.variables_, True)
        self.assertEqual(foo_key in mh.world_.variables_[bar_key].parent, True)
        self.assertEqual(bar_key in mh.world_.variables_[foo_key].children, True)
