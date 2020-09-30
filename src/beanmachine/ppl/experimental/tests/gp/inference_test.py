# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest
from functools import partial

import beanmachine.ppl as bm
import gpytorch
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.gp import likelihoods
from beanmachine.ppl.experimental.gp.kernels import PeriodicKernel, ScaleKernel
from beanmachine.ppl.experimental.gp.models import SimpleGP
from beanmachine.ppl.inference.single_site_no_u_turn_sampler import (
    SingleSiteNoUTurnSampler,
)
from gpytorch.distributions import MultivariateNormal


class Regression(SimpleGP):
    def __init__(self, x_train, y_train, kernel, likelihood, *args, **kwargs):
        mean = gpytorch.means.ConstantMean()
        super().__init__(x_train, y_train, mean, kernel, likelihood)

    def forward(self, data):
        if data.dim() > 2:
            data_shape = data.shape[1]
        else:
            data_shape = data.shape[0]
        jitter = torch.eye(data_shape, data_shape)
        for _ in range(data.dim() - 1):
            jitter = jitter.unsqueeze(0)
        mean = self.mean(data)
        cov = self.kernel(data) + jitter
        return MultivariateNormal(mean, cov)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        @bm.random_variable
        def outputscale_prior():
            return dist.Uniform(torch.tensor(1.0), torch.tensor(2.0))

        @bm.random_variable
        def lengthscale_prior():
            return dist.Uniform(torch.tensor([0.01]), torch.tensor([0.5]))

        @bm.random_variable
        def period_length_prior():
            return dist.Uniform(torch.tensor([0.05]), torch.tensor([2.5]))

        self.outputscale_prior = outputscale_prior
        self.lengthscale_prior = lengthscale_prior
        self.period_length_prior = period_length_prior

    def test_simple_regression(self):
        torch.manual_seed(1)
        n_samples = 100
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * (2 * math.pi))

        kernel = ScaleKernel(
            base_kernel=PeriodicKernel(
                period_length_prior=self.period_length_prior,
                lengthscale_prior=self.lengthscale_prior,
            ),
            outputscale_prior=self.outputscale_prior,
        )
        likelihood = likelihoods.GaussianLikelihood()

        gp = Regression(x, y, kernel, likelihood)
        gp_prior = partial(gp, x)
        obs = {gp.likelihood(gp_prior): y}
        queries = [
            self.outputscale_prior(),
            self.lengthscale_prior(),
            self.period_length_prior(),
        ]
        samples = SingleSiteNoUTurnSampler().infer(
            queries, obs, n_samples, num_chains=1
        )

        # get predictives
        x_test = torch.linspace(0, 1, 21).unsqueeze(-1)
        y_test = torch.sin(x_test * (2 * math.pi)).squeeze(0)
        gp.eval()
        s = samples.get_chain(0)  # noqa: E741
        lengthscale_samples = s[self.lengthscale_prior()]
        outputscale_samples = s[self.outputscale_prior()]
        period_length_samples = s[self.period_length_prior()]
        gp.bm_load_samples(
            {
                "kernel.outputscale": outputscale_samples,
                "kernel.base_kernel.lengthscale": lengthscale_samples.unsqueeze(-1),
                "kernel.base_kernel.period_length": period_length_samples.unsqueeze(-1),
            }
        )
        expanded_x_test = x_test.unsqueeze(0).repeat(n_samples, 1, 1)
        output = gp(expanded_x_test.detach())
        assert (y_test - output.mean.squeeze(0).mean(0)).abs().mean().item() < 1.0
