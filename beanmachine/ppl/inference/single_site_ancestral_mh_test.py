# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist


class SingleSiteAncestralMetropolisHastingsTest(unittest.TestCase):
    class SampleModel(object):
        @bm.random_variable
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @bm.random_variable
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    class ReproducibleModel(object):
        @bm.random_variable
        def K_minus_one(self):
            return dist.Poisson(rate=2.0)

        @bm.functional
        def K(self):
            return self.K_minus_one() + 1

        @bm.random_variable
        def mu(self):
            return dist.Normal(0, 1)

    def test_single_site_ancestral_mh(self):
        model = self.SampleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()
        foo_key = model.foo()
        bar_key = model.bar()
        mh.queries_ = [model.foo()]
        mh.observations_ = {model.bar(): torch.tensor(0.0)}
        mh._infer(10)
        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        world_vars = mh.world_.variables_.vars()
        self.assertEqual(foo_key in world_vars, True)
        self.assertEqual(bar_key in world_vars, True)
        self.assertEqual(foo_key in world_vars[bar_key].parent, True)
        self.assertEqual(bar_key in world_vars[foo_key].children, True)

    def test_single_site_ancestral_mh_reproducible_results(self):
        model = self.ReproducibleModel()
        mh = bm.SingleSiteAncestralMetropolisHastings()

        queries = [model.mu()]
        observations = {model.K(): torch.tensor(2.0)}

        torch.manual_seed(42)
        samples = mh.infer(queries, observations, num_samples=5, num_chains=1)
        run_1 = samples.get_variable(model.mu()).clone()

        torch.manual_seed(42)
        samples = mh.infer(queries, observations, num_samples=5, num_chains=1)
        run_2 = samples.get_variable(model.mu()).clone()
        self.assertTrue(run_1.allclose(run_2))

        torch.manual_seed(43)
        samples = mh.infer(queries, observations, num_samples=5, num_chains=1)
        run_3 = samples.get_variable(model.mu()).clone()
        self.assertFalse(run_1.allclose(run_3))
