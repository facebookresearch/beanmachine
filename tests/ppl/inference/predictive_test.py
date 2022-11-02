# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    def likelihood_dynamic(self, i):
        if self.likelihood_i(i).item() > 0:
            return dist.Normal(torch.zeros(1), torch.ones(1))
        else:
            return dist.Normal(5.0 * torch.ones(1), torch.ones(1))

    @bm.random_variable
    def prior_2(self):
        return dist.Uniform(torch.zeros(1, 2), torch.ones(1, 2))

    @bm.random_variable
    def likelihood_2(self, i):
        return dist.Bernoulli(self.prior_2())

    @bm.random_variable
    def likelihood_2_vec(self, i):
        return dist.Bernoulli(self.prior_2())

    @bm.random_variable
    def likelihood_reg(self):
        return dist.Normal(self.prior() * self.x, torch.tensor(1.0))

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
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=True)
        assert predictives[self.likelihood_i(0)].shape == (2, 10)
        assert predictives[self.likelihood_i(1)].shape == (2, 10)

    def test_posterior_predictive_seq(self):
        obs = {
            self.likelihood_i(0): torch.tensor(1.0),
            self.likelihood_i(1): torch.tensor(0.0),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior()], obs, num_samples=10, num_chains=2
        )
        assert post_samples[self.prior()].shape == (2, 10)
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=False)
        assert predictives[self.likelihood_i(0)].shape == (2, 10)
        assert predictives[self.likelihood_i(1)].shape == (2, 10)

    def test_predictive_dynamic(self):
        obs = {
            self.likelihood_dynamic(0): torch.tensor([0.9]),
            self.likelihood_dynamic(1): torch.tensor([4.9]),
        }
        # only query one of the variables
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior()], obs, num_samples=10, num_chains=2
        )
        assert post_samples[self.prior()].shape == (2, 10)
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=False)
        assert predictives[self.likelihood_dynamic(0)].shape == (2, 10)
        assert predictives[self.likelihood_dynamic(1)].shape == (2, 10)

    def test_predictive_data(self):
        x = torch.randn(4)
        y = torch.randn(4) + 2.0
        self.x = x
        obs = {self.likelihood_reg(): y}
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior()], obs, num_samples=10, num_chains=2
        )
        assert post_samples[self.prior()].shape == (2, 10)
        self.x = torch.randn(4, 1, 1)
        test_query = self.likelihood_reg()
        predictives = bm.simulate([test_query], post_samples, vectorized=True)
        assert predictives[test_query].shape == (4, 2, 10)

    def test_posterior_predictive_1d(self):
        obs = {self.likelihood_1(): torch.tensor([1.0])}
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior_1()], obs, num_samples=10, num_chains=1
        )
        assert post_samples[self.prior_1()].shape == (1, 10, 1)
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=True)
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
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=True)
        predictive_0 = predictives[self.likelihood_2(0)]
        predictive_1 = predictives[self.likelihood_2(1)]
        assert predictive_0.shape == (2, 10, 1, 2)
        assert predictive_1.shape == (2, 10, 1, 2)
        assert (predictive_1 - predictive_0).sum().item() != 0

    def test_empirical(self):
        obs = {
            self.likelihood_i(0): torch.tensor(1.0),
            self.likelihood_i(1): torch.tensor(0.0),
            self.likelihood_i(2): torch.tensor(0.0),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior()], obs, num_samples=10, num_chains=4
        )
        empirical = bm.empirical([self.prior()], post_samples, num_samples=26)
        assert empirical[self.prior()].shape == (1, 26)
        predictives = bm.simulate(list(obs.keys()), post_samples, vectorized=True)
        empirical = bm.empirical(list(obs.keys()), predictives, num_samples=27)
        assert len(empirical) == 3
        assert empirical[self.likelihood_i(0)].shape == (1, 27)
        assert empirical[self.likelihood_i(1)].shape == (1, 27)

    def test_return_inference_data(self):
        torch.manual_seed(10)
        obs = {
            self.likelihood_2(0): torch.tensor([[1.0, 1.0]]),
            self.likelihood_2(1): torch.tensor([[0.0, 1.0]]),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior_2()], obs, num_samples=10, num_chains=2
        )

        assert post_samples[self.prior_2()].shape == (2, 10, 1, 2)
        predictives = bm.simulate(
            list(obs.keys()),
            post_samples,
            vectorized=True,
        ).to_inference_data()
        assert "posterior" in predictives
        assert "observed_data" in predictives
        assert "log_likelihood" in predictives
        assert "posterior_predictive" in predictives
        assert predictives.posterior_predictive[self.likelihood_2(0)].shape == (
            2,
            10,
            1,
            2,
        )
        assert predictives.posterior_predictive[self.likelihood_2(1)].shape == (
            2,
            10,
            1,
            2,
        )

    def test_posterior_dict(self):
        obs = {
            self.likelihood_i(0): torch.tensor(1.0),
            self.likelihood_i(1): torch.tensor(0.0),
        }

        posterior = {self.prior(): torch.tensor([0.5, 0.5])}

        predictives_dict = bm.simulate(list(obs.keys()), posterior)
        assert predictives_dict[self.likelihood_i(0)].shape == (1, 2)
        assert predictives_dict[self.likelihood_i(1)].shape == (1, 2)

    def test_posterior_dict_predictive(self):
        obs = {
            self.likelihood_i(0): torch.tensor(1.0),
            self.likelihood_i(1): torch.tensor(0.0),
        }
        post_samples = bm.SingleSiteAncestralMetropolisHastings().infer(
            [self.prior()], obs, num_samples=10, num_chains=1
        )
        assert post_samples[self.prior()].shape == (1, 10)

        post_samples_dict = dict(post_samples)
        predictives_dict = bm.simulate(list(obs.keys()), post_samples_dict)
        assert predictives_dict[self.likelihood_i(0)].shape == (1, 10)
        assert predictives_dict[self.likelihood_i(1)].shape == (1, 10)
