# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
from beanmachine.ppl.inference.single_site_random_walk import SingleSiteRandomWalk
from beanmachine.ppl.model.statistical_model import sample


# A distribution which apparently takes values on the full number line,
# but in reality it only returns zero when sampled from.
class RealSupportDist(dist.Distribution):
    has_enumerate_support = False
    support = dist.constraints.real
    has_rsample = True

    # Ancestral sampling will only return zero.
    def rsample(self, sample_shape):
        return torch.zeros(sample_shape)

    # Not a properly defined PDF on the full support, but allows MCMC to explore.
    def log_prob(self, value):
        return torch.zeros(value.shape)


# A distribution which apparently takes values on the non-negative integers,
# but in reality it only returns zero when sampled from.
class IntegerSupportDist(dist.Distribution):
    has_enumerate_support = False
    support = dist.constraints.integer_interval(0, 100)
    has_rsample = True

    # Ancestral sampling will only return zero.
    def rsample(self, sample_shape):
        return torch.zeros(sample_shape)

    # Not a properly defined PDF on the full support, but allows MCMC to explore.
    def log_prob(self, value):
        return torch.zeros(value.shape)


class SingleSiteRandomWalkTest(unittest.TestCase):
    class SampleModel(object):
        @sample
        def foo(self):
            return dist.Normal(torch.tensor(0.0), torch.tensor(1.0))

        @sample
        def bar(self):
            return dist.Normal(self.foo(), torch.tensor(1.0))

    def test_single_site_random_walk_mechanics(self):
        model = self.SampleModel()
        mh = SingleSiteRandomWalk()
        foo_key = model.foo()
        bar_key = model.bar()
        mh.queries_ = [foo_key]
        mh.observations_ = {model.bar(): torch.tensor(0.0)}
        mh._infer(10)
        # using _infer instead of infer, as world_ would be reset at the end
        # infer
        self.assertIn(foo_key, mh.world_.variables_)
        self.assertIn(bar_key, mh.world_.variables_)
        self.assertIn(foo_key, mh.world_.variables_[bar_key].parent)
        self.assertIn(bar_key, mh.world_.variables_[foo_key].children)

    class RealSupportModel(object):
        @sample
        def p(self):
            return RealSupportDist()

        @sample
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    class IntegerSupportModel(object):
        @sample
        def p(self):
            return IntegerSupportDist()

        @sample
        def q(self):
            return dist.Normal(self.p().to(dtype=torch.float32), torch.tensor(1.0))

    # Check that RW sampler is used when it should be.
    def test_single_site_random_walk_support(self):
        model = self.RealSupportModel()
        mh = SingleSiteRandomWalk()
        p_key = model.p()
        queries = [p_key]
        observations = {model.q(): torch.tensor(100.0)}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        """
        If the ancestral sampler is used, then every sample
        drawn from the chain will be 0. This is by true by
        the construction of the rsample function.
        Conversely, normal noise != 0 w.p. 1, giving some sample which != 0.
        For RealSupportModel, we expect the RW sampler to be used.
        """
        self.assertIn(False, [0 == pred for pred in predictions])

    class NormalNormalModel(object):
        def __init__(self, ndim=1):
            self.ndim = ndim

        @sample
        def p(self):
            return dist.Normal(torch.zeros(self.ndim), torch.ones(self.ndim))

        @sample
        def q(self):
            return dist.Normal(self.p(), torch.ones(self.ndim))

    def test_single_site_random_walk_rate(self):
        model = self.NormalNormalModel(1)
        mh = SingleSiteRandomWalk(step_size=10)
        p_key = model.p()
        queries = [p_key]
        observations = {model.q(): torch.tensor(100.0)}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        self.assertIn(True, [45 < pred < 55 for pred in predictions])

    def test_single_site_random_walk_rate_vector(self):
        model = self.NormalNormalModel(2)
        mh = SingleSiteRandomWalk(step_size=10)
        p_key = model.p()
        queries = [p_key]
        observations = {model.q(): torch.tensor([100.0, -100.0])}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        self.assertIn(True, [45 < pred[0] < 55 for pred in predictions])
        self.assertIn(True, [-55 < pred[1] < -45 for pred in predictions])
