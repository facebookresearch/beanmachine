# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.examples.conjugate_models import (
    BetaBinomialModel,
    CategoricalDirichletModel,
    GammaNormalModel,
    NormalNormalModel,
)
from beanmachine.ppl.inference.single_site_random_walk import (
    SingleSiteRandomWalk,
)


# A distribution which apparently takes values on the full number line,
# but in reality it only returns zero when sampled from.
class RealSupportDist(dist.Distribution):
    has_enumerate_support = False
    support = dist.constraints.real
    has_rsample = True
    arg_constraints = {}

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
    arg_constraints = {}

    # Ancestral sampling will only return one.
    def rsample(self, sample_shape):
        return torch.ones(sample_shape)

    # Not a properly defined PDF on the full support, but allows MCMC to explore.
    def log_prob(self, value):
        return torch.zeros(value.shape)


# A distribution which apparently takes values on an interval of the number line,
# but in reality it only returns 1 when sampled from.
class IntervalRealSupportDist(dist.Distribution):
    has_enumerate_support = False
    support = dist.constraints.interval(lower_bound=2.0, upper_bound=20.0)
    has_rsample = True
    arg_constraints = {}

    # Ancestral sampling will only return zero.
    def rsample(self, sample_shape):
        return 3 * torch.ones(sample_shape)

    # Not a properly defined PDF on the full support, but allows MCMC to explore.
    def log_prob(self, value):
        return torch.zeros(value.shape)


# A distribution which apparently takes values on the non-negative integers,
# but in reality it only returns zero when sampled from.
class IntegerSupportDist(dist.Distribution):
    has_enumerate_support = False
    support = dist.constraints.integer_interval(0.0, 100.0)
    has_rsample = True
    arg_constraints = {}

    # Ancestral sampling will only return zero.
    def rsample(self, sample_shape):
        return torch.zeros(sample_shape)

    # Not a properly defined PDF on the full support, but allows MCMC to explore.
    def log_prob(self, value):
        return torch.zeros(value.shape)


class SingleSiteRandomWalkTest(unittest.TestCase):
    """
    These tests test for the control flow which branches
    based on node_distribution.support
    """

    class RealSupportModel(object):
        @bm.random_variable
        def p(self):
            return RealSupportDist()

        @bm.random_variable
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    class HalfRealSupportModel(object):
        @bm.random_variable
        def p(self):
            return HalfRealSupportDist()

        @bm.random_variable
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    class IntervalRealSupportModel(object):
        def __init__(self):
            self.lower_bound = IntervalRealSupportDist().support.lower_bound
            self.upper_bound = IntervalRealSupportDist().support.upper_bound

        @bm.random_variable
        def p(self):
            return IntervalRealSupportDist()

        @bm.random_variable
        def q(self):
            return dist.Normal(self.p(), torch.tensor(1.0))

    class IntegerSupportModel(object):
        @bm.random_variable
        def p(self):
            return IntegerSupportDist()

        @bm.random_variable
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

    def test_single_site_random_walk_interval_support(self):
        lower_bound = IntervalRealSupportDist().support.lower_bound
        upper_bound = IntervalRealSupportDist().support.upper_bound

        # Test for a single item of evidence
        def inner_fnc(evidence: float):
            model = self.IntervalRealSupportModel()
            mh = SingleSiteRandomWalk()
            p_key = model.p()
            queries = [p_key]
            observations = {model.q(): evidence.detach().clone()}
            predictions = mh.infer(queries, observations, 20)
            predictions = predictions.get_chain()[p_key]
            """
            All generated samples should remain in the correct support
            if the transform is computed properly
            """

            self.assertNotIn(
                False, [lower_bound <= pred <= upper_bound for pred in predictions]
            )

        # We're mostly interested in the boundary cases
        evidences = torch.cat(
            (
                torch.linspace(lower_bound + 0.1, lower_bound + 1, 4),
                torch.linspace(upper_bound - 1, upper_bound - 0.1, 4),
            )
        )

        for e in evidences:
            inner_fnc(e)

    """
    Adaptive
    """

    def test_single_site_adaptive_random_walk(self):
        model = NormalNormalModel(
            mu=torch.zeros(1), std=torch.ones(1), sigma=torch.ones(1)
        )
        mh = SingleSiteRandomWalk(step_size=4)
        p_key = model.normal_p()
        queries = [p_key]
        observations = {model.normal(): torch.tensor(100.0)}
        predictions = mh.infer(queries, observations, 100, 30)
        predictions = predictions.get_chain()[p_key]
        self.assertIn(True, [45 < pred < 55 for pred in predictions])

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
        mh = SingleSiteRandomWalk(step_size=4.0)
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

    def test_single_site_random_walk_interval_support_rate(self):
        model = BetaBinomialModel(
            alpha=torch.ones(1) * 2.0, beta=torch.ones(1), n=torch.ones(1) * 10.0
        )
        mh = SingleSiteRandomWalk(step_size=0.3)
        p_key = model.theta()
        queries = [p_key]
        observations = {model.x(): torch.tensor([10.0])}
        predictions = mh.infer(queries, observations, 50)
        predictions = predictions.get_chain()[p_key]
        """
        Our single piece of evidence is the observed value 10.
        This is a large observation w.r.t our model  . This
        implies that the Binomial distirubtion has very large parameter p, so
        samples from the Beta distribution will have similarly large values in
        expectation. For RWMH with small step size, we expect to accept enough
        proposals to reach this value in < 50 steps.
        """
        self.assertIn(True, [pred > 0.9 for pred in predictions])

    def test_single_site_random_walk_simplex_support_rate(self):
        model = CategoricalDirichletModel(alpha=torch.tensor([1.0, 10.0]))
        mh = SingleSiteRandomWalk(step_size=1.0)
        p_key = model.dirichlet()
        queries = [p_key]
        observations = {model.categorical(): torch.tensor([1.0, 1.0, 1.0])}
        predictions = mh.infer(queries, observations, 50)
        predictions = predictions.get_chain()[p_key]
        """
        Our single piece of evidence is the observed value 1.
        This is a large observation w.r.t the simplex, which has interval [0,1].
        Based on our model, we expect that this evidence is drawn from
        category 1 rather than category 0. So pred[0] << pred[1] typically.
        """
        self.assertIn(True, [pred[0] < 0.1 for pred in predictions])
