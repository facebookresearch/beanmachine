# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import beanmachine.ppl as bm
import torch
import torch.distributions as dist
from beanmachine.contrib.gp import kernels, likelihoods
from beanmachine.contrib.gp.models import BoTorchGP, SimpleGP
from beanmachine.ppl.inference.single_site_no_u_turn_sampler import (
    SingleSiteNoUTurnSampler,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior


class ModelTest(unittest.TestCase):
    def setUp(self):
        @bm.random_variable
        def p():
            return dist.HalfNormal(torch.tensor(1.0))

        @bm.random_variable
        def mean():
            return dist.Normal(torch.zeros(1), torch.ones(1))

        self.p = p
        x = torch.randn(3, 1)
        y = torch.randn(3)
        l = likelihoods.GaussianLikelihood()  # noqa: E741
        kernel = kernels.MaternKernel(lengthscale_prior=p)
        self.model = SimpleGP(x, y, mean, kernel, l)
        self.bo_model = BoTorchGP(x, y, mean, kernel, l)

    def test_infer(self):
        self.model.train()
        SingleSiteNoUTurnSampler().infer([self.p()], {}, num_samples=2, num_chains=1)

    def test_load_and_predict(self):
        self.model.eval()
        d = {"kernel.lengthscale": torch.ones(1), "mean": torch.tensor(1.0)}
        self.model.bm_load_samples(d)
        assert self.model.kernel.lengthscale.item() == 1.0
        assert isinstance(self.model(torch.randn(3, 1)), dist.MultivariateNormal)

    def test_posterior(self):
        self.bo_model.eval()
        d = {"kernel.lengthscale": torch.ones(1), "mean": torch.tensor(1.0)}
        self.bo_model.bm_load_samples(d)
        assert isinstance(self.bo_model.posterior(torch.randn(3, 1)), GPyTorchPosterior)
        obs_noise = torch.ones(1, 1)
        mvn = self.bo_model.posterior(torch.randn(3, 1), obs_noise)
        assert isinstance(mvn, GPyTorchPosterior)
