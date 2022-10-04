# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.gp import (
    bm_sample_from_prior,
    make_prior_random_variables,
)
from beanmachine.ppl.experimental.gp.models import BoTorchGP, SimpleGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import kernels, likelihoods
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior, UniformPrior


class ModelTest(unittest.TestCase):
    def setUp(self):
        x = torch.randn(3, 1)
        y = torch.randn(3)
        mean = ConstantMean(constant_prior=UniformPrior(-1, 1))
        kernel = kernels.MaternKernel(lengthscale_prior=GammaPrior(0.5, 0.5))
        lik = likelihoods.GaussianLikelihood()
        self.model = SimpleGP(x, y, mean, kernel, lik)
        self.bo_model = BoTorchGP(x, y, mean, kernel, lik)

        self.name_to_rv = make_prior_random_variables(self.model)

        @bm.random_variable
        def y():
            sampled_model = bm_sample_from_prior(
                self.model.to_pyro_random_module(),
                self.name_to_rv,
            )
            return sampled_model.likelihood(sampled_model(x))

        self.y = y

    def test_infer(self):
        self.model.train()
        bm.SingleSiteNoUTurnSampler().infer(
            list(self.name_to_rv.values()), {}, num_samples=2, num_chains=1
        )

    def test_load_and_predict(self):
        self.model.eval()
        d = {
            "kernel.lengthscale_prior": torch.ones(1),
            "mean.mean_prior": torch.tensor(1.0),
        }
        self.model.bm_load_samples(d)
        assert self.model.kernel.lengthscale.item() == 1.0
        assert isinstance(self.model(torch.randn(3, 1)), dist.MultivariateNormal)

    def test_posterior(self):
        self.bo_model.eval()
        d = {
            "kernel.lengthscale_prior": torch.ones(1),
            "mean.mean_prior": torch.tensor(1.0),
        }
        self.bo_model.bm_load_samples(d)
        assert isinstance(self.bo_model.posterior(torch.randn(3, 1)), GPyTorchPosterior)
        obs_noise = torch.ones(1, 1)
        mvn = self.bo_model.posterior(torch.randn(3, 1), obs_noise)
        assert isinstance(mvn, GPyTorchPosterior)
