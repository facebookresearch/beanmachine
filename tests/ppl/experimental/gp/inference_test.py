# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import beanmachine.ppl as bm
import gpytorch
import torch
from beanmachine.ppl.experimental.gp import (
    bm_sample_from_prior,
    make_prior_random_variables,
)
from beanmachine.ppl.experimental.gp.models import SimpleGP
from gpytorch import likelihoods
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import PeriodicKernel, ScaleKernel
from gpytorch.priors import UniformPrior


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
        if cov.dim() > mean.dim() + 1:
            cov = cov.squeeze(0)
        return MultivariateNormal(mean, cov)


class InferenceTests(unittest.TestCase):
    def test_simple_regression(self):
        torch.manual_seed(1)
        n_samples = 100
        x_train = torch.linspace(0, 1, 10)
        y_train = torch.sin(x_train * (2 * math.pi))

        kernel = ScaleKernel(
            base_kernel=PeriodicKernel(
                period_length_prior=UniformPrior(0.5, 1.5),
                lengthscale_prior=UniformPrior(0.01, 1.5),
            ),
            outputscale_prior=UniformPrior(0.01, 2.0),
        )
        likelihood = likelihoods.GaussianLikelihood()
        likelihood.noise = 1e-4

        gp = Regression(x_train, y_train, kernel, likelihood)
        name_to_rv = make_prior_random_variables(gp)

        @bm.random_variable
        def y():
            sampled_model = bm_sample_from_prior(gp.to_pyro_random_module(), name_to_rv)
            return sampled_model.likelihood(sampled_model(x_train))

        queries = list(name_to_rv.values())
        obs = {y(): y_train}
        samples = bm.GlobalNoUTurnSampler(nnc_compile=False).infer(
            queries, obs, n_samples, num_chains=1
        )

        # get predictives
        x_test = torch.linspace(0, 1, 21).unsqueeze(-1)
        y_test = torch.sin(x_test * (2 * math.pi)).squeeze(0)
        gp.eval()
        s = samples.get_chain(0)
        lengthscale_samples = s[name_to_rv["kernel.base_kernel.lengthscale_prior"]]
        outputscale_samples = s[name_to_rv["kernel.outputscale_prior"]]
        period_length_samples = s[name_to_rv["kernel.base_kernel.period_length_prior"]]
        gp.pyro_load_from_samples(
            {
                "kernel.outputscale_prior": outputscale_samples,
                "kernel.base_kernel.lengthscale_prior": lengthscale_samples,
                "kernel.base_kernel.period_length_prior": period_length_samples,
            }
        )
        expanded_x_test = x_test.unsqueeze(0).repeat(n_samples, 1, 1)
        output = gp.likelihood(gp(expanded_x_test.detach()))
        assert (
            (y_test.squeeze() - output.mean.squeeze().mean(0)).abs().mean() < 1.0
        ).item()
