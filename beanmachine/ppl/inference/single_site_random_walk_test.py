# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
import torch.distributions as dist
from beanmachine.ppl.examples.conjugate_models import (
    GammaNormalModel,
    NormalNormalModel,
)
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


# A distribution which apparently takes values on the non-negative number line,
# but in reality it only returns 1 when sampled from.
class HalfRealSupportDist(dist.Distribution):
    has_enumerate_support = False
    support = dist.constraints.greater_than(lower_bound=0.0)
    has_rsample = True

    # Ancestral sampling will only return one.
    def rsample(self, sample_shape):
        return torch.ones(sample_shape)

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

    """
    These tests test for the control flow which branches
    based on node_distribution.support
    """

    class RealSupportModel(object):
        @sample
        def p(self):
            return RealSupportDist()

        @sample
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    class HalfRealSupportModel(object):
        @sample
        def p(self):
            return HalfRealSupportDist()

        @sample
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    class IntegerSupportModel(object):
        @sample
        def p(self):
            return IntegerSupportDist()

        @sample
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    def test_single_site_random_walk_full_support(self):
        model = self.RealSupportModel()
        mh = SingleSiteRandomWalk()
        p_key = model.p()
        queries = [p_key]
        observations = {model.q(): torch.tensor(1.0)}
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

    def test_single_site_random_walk_half_support(self):
        model = self.HalfRealSupportModel()
        mh = SingleSiteRandomWalk()
        p_key = model.p()
        queries = [p_key]
        observations = {model.q(): torch.tensor(100.0)}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        # Discard the first sample, it may not be drawn from the node's distribution
        predictions = predictions[1:]
        """
        If the ancestral sampler is used, then every sample
        drawn from the chain will be 1. This is by true by
        the construction of the rsample function.
        If RW is correctly reached by control flow, then rsample will
        draw from a Gamma distribution.
        """
        self.assertIn(False, [pred == 1 for pred in predictions])

    """
    These tests test for quick approximate convergence in conjugate models.
    """

    def test_single_site_random_walk_rate(self):
        model = NormalNormalModel(
            mu=torch.zeros(1), std=torch.ones(1), sigma=torch.ones(1)
        )
        mh = SingleSiteRandomWalk(step_size=10)
        p_key = model.normal_p()
        queries = [p_key]
        observations = {model.normal(): torch.tensor(100.0)}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        self.assertIn(True, [45 < pred < 55 for pred in predictions])

    def test_single_site_random_walk_rate_vector(self):
        model = NormalNormalModel(
            mu=torch.zeros(2), std=torch.ones(2), sigma=torch.ones(2)
        )
        mh = SingleSiteRandomWalk(step_size=10)
        p_key = model.normal_p()
        queries = [p_key]
        observations = {model.normal(): torch.tensor([100.0, -100.0])}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        self.assertIn(True, [45 < pred[0] < 55 for pred in predictions])
        self.assertIn(True, [-55 < pred[1] < -45 for pred in predictions])

    def test_single_site_random_walk_half_support_rate(self):
        model = GammaNormalModel(
            shape=torch.ones(1), rate=torch.ones(1), mu=torch.ones(1)
        )
        mh = SingleSiteRandomWalk(step_size=3.0)
        p_key = model.gamma()
        queries = [p_key]
        observations = {model.normal(): torch.tensor([100.0])}
        predictions = mh.infer(queries, observations, 100)
        predictions = predictions.get_chain()[p_key]
        """
        Our single piece of evidence is the observed value 100.
        100 is a very large observation w.r.t our model of mu = 1. This
        implies that the normal distirubtion has very high variance, so samples
        from the Gamma distribution will have very small values in expectation.
        For RWMH with large step size, we expect to see this in < 100 steps.
        """
        self.assertIn(True, [pred < 0.01 for pred in predictions])
