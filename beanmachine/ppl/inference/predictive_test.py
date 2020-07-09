# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist


class PredictiveTest(unittest.TestCase):
    @bm.random_variable
    def prior(self):
        return dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))

    @bm.random_variable
    def likelihood(self):
        return dist.Bernoulli(self.prior())

    @bm.random_variable
    def likelihood_i(self, i):
        return dist.Bernoulli(self.prior())

    @bm.random_variable
    def prior_1(self):
        return dist.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

    @bm.random_variable
    def likelihood_1(self):
        return dist.Bernoulli(self.prior_1())

    @bm.random_variable
    def prior_2(self):
        return dist.Uniform(torch.zeros(1, 2), torch.ones(1, 2))

    @bm.random_variable
    def likelihood_2(self, i):
        return dist.Bernoulli(self.prior_2())

    @bm.random_variable
    def likelihood_2_vec(self, i):
        return dist.Bernoulli(self.prior_2())

    def test_prior_predictive(self):
        queries = [self.prior(), self.likelihood()]
        predictives = bm.simulate(queries, num_samples=10)
        assert predictives[self.prior()].shape == (1, 10)
        assert predictives[self.likelihood()].shape == (1, 10)

    def test_posterior_predictive(self):
        obs = {
            self.likelihood_i(0): torch.tensor(1.0),
            self.likelihood_i(1): torch.tensor(0.0),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior()], obs, num_samples=10, num_chains=2
        )
        assert post_samples[self.prior()].shape == (2, 10)
        predictives = bm.simulate(list(obs.keys()), post_samples)
        predictives[self.likelihood_i(0)].shape == (2, 10)
        assert predictives[self.likelihood_i(1)].shape == (2, 10)

    def test_posterior_predictive_1d(self):
        obs = {self.likelihood_1(): torch.tensor([1.0])}
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior_1()], obs, num_samples=10, num_chains=1
        )
        assert post_samples[self.prior_1()].shape == (1, 10, 1)
        predictives = bm.simulate(list(obs.keys()), post_samples)
        y = predictives[self.likelihood_1()].shape
        assert y == (1, 10, 1)

    def test_multi_chain_infer_predictive_2d(self):
        torch.manual_seed(10)
        obs = {
            self.likelihood_2(0): torch.tensor([[1.0, 1.0]]),
            self.likelihood_2(1): torch.tensor([[0.0, 1.0]]),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior_2()], obs, num_samples=10, num_chains=2
        )

        assert post_samples[self.prior_2()].shape == (2, 10, 1, 2)
        predictives = bm.simulate(list(obs.keys()), post_samples)
        predictive_0 = predictives[self.likelihood_2(0)]
        predictive_1 = predictives[self.likelihood_2(1)]
        assert predictive_0.shape == (2, 10, 1, 2)
        assert predictive_1.shape == (2, 10, 1, 2)
        assert (predictive_1 - predictive_0).sum().item() != 0
