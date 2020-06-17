# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.examples.conjugate_models import NormalNormalModel
from beanmachine.ppl.experimental.inference_compilation.ic_infer import ICInference
from torch import tensor


class InferenceCompilationTest(unittest.TestCase):
    class WaldsModel:
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

    def test_walds_identity(self):
        model = self.WaldsModel()
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
