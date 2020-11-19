# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.distribution import Flat
from beanmachine.ppl.examples.conjugate_models import NormalNormalModel
from beanmachine.ppl.experimental.inference_compilation.ic_infer import ICInference
from torch import tensor


class InferenceCompilationTest(unittest.TestCase):
    class RandomGaussianSum:
        @bm.random_variable
        def N(self):
            k = 10
            return dist.Categorical(probs=torch.ones(k) * 1.0 / k)

        @bm.random_variable
        def x(self, i, j):
            return dist.Normal(loc=tensor(1.0), scale=tensor(1.0))

        @bm.random_variable
        def S(self, j):
            "The Markov blanket of S varies depending on the value of N"
            loc = sum(self.x(i, j) for i in range(self.N().int().item()))
            return dist.Normal(loc=loc, scale=tensor(1.0))

    class GMM:
        def __init__(self, K=2, d=1):
            self.K = K
            self.d = d

        @bm.random_variable
        def mu(self, j):
            if self.d == 1:
                return dist.Normal(0, 1)
            else:
                return dist.MultivariateNormal(
                    loc=torch.zeros(self.d), covariance_matrix=torch.eye(self.d)
                )

        @bm.random_variable
        def c(self, i):
            return dist.Categorical(probs=torch.ones(self.K) * 1.0 / self.K)

        @bm.random_variable
        def x(self, i):
            c = self.c(i).int().item()
            mu = self.mu(c)
            if self.d == 1:
                return dist.Normal(mu, 0.1)
            else:
                return dist.MultivariateNormal(
                    loc=mu, covariance_matrix=0.1 * torch.eye(self.d)
                )

    class IndependentBernoulliGaussian:
        @bm.random_variable
        def foo(self):
            return dist.Normal(0, 1)

        @bm.random_variable
        def bar(self, i):
            return dist.Bernoulli(0.5)

    def setUp(self):
        torch.manual_seed(42)

    def test_normal_normal(self):
        prior_mean = -1.0
        model = NormalNormalModel(
            mu=tensor(prior_mean), std=tensor(1.0), sigma=tensor(1.0)
        )

        observed_value = prior_mean - 1.0
        observations = {model.normal(): tensor(observed_value)}
        ic = ICInference()
        ic.compile(observations.keys())
        queries = [model.normal_p()]
        samples = ic.infer(queries, observations, num_samples=1500, num_chains=1)
        assert samples[model.normal_p()].mean().item() <= prior_mean
        assert samples[model.normal_p()].mean().item() >= observed_value

    def test_random_sum(self):
        model = self.RandomGaussianSum()
        ic = ICInference()
        observations = {model.S(0): tensor(1.8), model.S(1): tensor(2.2)}
        ic.compile(observations.keys(), num_worlds=100)
        queries = [model.N()]
        samples = ic.infer(queries, observations, num_samples=100, num_chains=1)

        # observations likelihood (by Wald's identity) maximized at N=2, so
        # posterior mean should be below prior mean E[N] = 4.5
        N_posterior_mean_estimate = samples[model.N()].float().mean().item()
        assert (
            N_posterior_mean_estimate < 4.5
        ), f"Expected {N_posterior_mean_estimate} < 4.5"

    def test_gmm(self):
        model = self.GMM(K=2)
        ic = ICInference()
        observations = {
            model.x(0): tensor(1.0),
            model.x(1): tensor(-1.0),
            model.x(2): tensor(1.0),
            model.x(3): tensor(-1.0),
        }
        ic.compile(observations.keys(), num_worlds=100)
        queries = [model.mu(i) for i in range(model.K)]
        ic_samples = ic.infer(queries, observations, num_samples=200, num_chains=1)

        posterior_means_mu = bm.Diagnostics(ic_samples).summary()["avg"]
        self.assertAlmostEqual(posterior_means_mu.min(), -1, delta=0.3)
        self.assertAlmostEqual(posterior_means_mu.max(), 1, delta=0.3)

    def test_gmm_random_rvidentifier_embeddings(self):
        model = self.GMM(K=2)
        ic = ICInference()
        observations = {
            model.x(0): tensor(1.0),
            model.x(1): tensor(-1.0),
            model.x(2): tensor(1.0),
            model.x(3): tensor(-1.0),
        }
        ic.compile(observations.keys(), num_worlds=100, node_id_embedding_dim=32)
        queries = [model.mu(i) for i in range(model.K)]
        ic_samples = ic.infer(queries, observations, num_samples=200, num_chains=1)

        posterior_means_mu = bm.Diagnostics(ic_samples).summary()["avg"]
        self.assertAlmostEqual(posterior_means_mu.min(), -1, delta=0.3)
        self.assertAlmostEqual(posterior_means_mu.max(), 1, delta=0.3)

    def test_gmm_2d(self):
        model = self.GMM(K=2, d=2)
        ic = ICInference()
        observations = {
            model.x(0): tensor([0.0, 1.0]),
            model.x(1): tensor([0.0, -1.0]),
            model.x(2): tensor([0.0, 1.0]),
            model.x(3): tensor([0.0, -1.0]),
        }
        ic.compile(observations.keys(), num_worlds=500)
        queries = [model.mu(i) for i in range(model.K)]
        ic_samples = ic.infer(queries, observations, num_samples=300, num_chains=1)

        posterior_means_mu = bm.Diagnostics(ic_samples).summary()["avg"]
        self.assertAlmostEqual(posterior_means_mu.min(), -1, delta=0.3)
        self.assertAlmostEqual(posterior_means_mu.max(), 1, delta=0.3)

    def test_raises_on_matrix_distributions(self):
        rv = bm.random_variable(lambda: Flat((2, 2)))
        rv_obs = bm.random_variable(lambda: dist.Normal(rv()[0, 0], 1))
        with self.assertRaises(NotImplementedError):
            ICInference().compile([rv_obs()])

    def test_do_adaptation(self):
        # undertrained model
        model = NormalNormalModel(mu=tensor(0.0), std=tensor(1.0), sigma=tensor(1.0))
        observed_value = -1.0
        observations = {model.normal(): tensor(observed_value)}
        ic = ICInference()
        ic.compile(observations.keys(), num_worlds=20)

        node = model.normal_p()
        ic.queries_ = [node]
        ic.observations_ = observations
        ic.initialize_world(initialize_from_prior=True)

        world = ic.world_
        node_var = world.get_node_in_world_raise_error(node)

        # draw some posterior samples to compute empirical KL divergence over
        samples = ic._infer(num_samples=10)

        # compute empirical inclusive KL before adaptation
        ic_proposer = ic._proposers(node)
        before_adaptation_kldiv = -(
            ic_proposer.get_proposal_distribution(node, node_var, world, {})[0]
            .proposal_distribution.log_prob(samples[node])
            .sum()
        )

        # run adaptation
        ic._infer(num_samples=0, num_adaptive_samples=100)

        # compute empirical inclusive KL after adaptation
        ic_proposer = ic._proposers(node)
        after_adaptation_kldiv = -(
            ic_proposer.get_proposal_distribution(node, node_var, world, {})[0]
            .proposal_distribution.log_prob(samples[node])
            .sum()
        )

        assert before_adaptation_kldiv > after_adaptation_kldiv

    def test_independent_bernoulli_gaussian(self):
        # this test exercises the following edge cases:
        #  * IC on a leaf node with no latent parents
        #  * IC proposer for Bernoulli
        #  * Querying nodes not provided to `compile()`
        model = self.IndependentBernoulliGaussian()
        ic = ICInference()
        ic.compile([model.bar(10)], num_worlds=10)

        ic.infer(
            [model.foo()] + [model.bar(i) for i in range(100)],
            {model.bar(10): torch.tensor(1.0)},
            num_samples=100,
            num_chains=1,
        )

    def test_multiple_component_1d_gmm_density_estimator(self):
        model = self.GMM(K=2)
        ic = ICInference()
        observations = {
            model.x(0): tensor(1.0),
            model.x(1): tensor(-1.0),
            model.x(2): tensor(1.0),
            model.x(3): tensor(-1.0),
        }
        ic.compile(observations.keys(), num_worlds=100, gmm_num_components=3)
        queries = [model.mu(i) for i in range(model.K)]
        ic_samples = ic.infer(queries, observations, num_samples=200, num_chains=1)

        posterior_means_mu = bm.Diagnostics(ic_samples).summary()["avg"]
        self.assertAlmostEqual(posterior_means_mu.min(), -1, delta=0.3)
        self.assertAlmostEqual(posterior_means_mu.max(), 1, delta=0.3)

    def test_multiple_component_2d_gmm_density_estimator(self):
        model = self.GMM(K=2, d=2)
        ic = ICInference()
        observations = {
            model.x(0): tensor([0.0, 1.0]),
            model.x(1): tensor([0.0, -1.0]),
            model.x(2): tensor([0.0, 1.0]),
            model.x(3): tensor([0.0, -1.0]),
        }
        ic.compile(observations.keys(), num_worlds=500, gmm_num_components=3)
        queries = [model.mu(i) for i in range(model.K)]
        ic_samples = ic.infer(queries, observations, num_samples=300, num_chains=1)

        posterior_means_mu = bm.Diagnostics(ic_samples).summary()["avg"]
        self.assertAlmostEqual(posterior_means_mu.min(), -1, delta=0.3)
        self.assertAlmostEqual(posterior_means_mu.max(), 1, delta=0.3)
