# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest
from functools import partial, reduce

import beanmachine.ppl as bm
import gpytorch
import gpytorch.distributions as gdist
import torch
import torch.distributions as dist
from beanmachine.ppl.experimental.gp import likelihoods
from beanmachine.ppl.experimental.gp.kernels import (
    Kernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)
from beanmachine.ppl.inference.single_site_ancestral_mh import (
    SingleSiteAncestralMetropolisHastings,
)
from beanmachine.ppl.inference.single_site_hamiltonian_monte_carlo import (
    SingleSiteHamiltonianMonteCarlo,
)


class StructureAdditiveKernel(Kernel):
    def __init__(self, num_contexts, lengthscale_prior, outputscale_prior):
        super().__init__()
        self.num_contexts = num_contexts

        self.base_kernel = MaternKernel(lengthscale_prior=lengthscale_prior)

        self.kernels = []
        for i in range(num_contexts):
            per_context_prior = partial(outputscale_prior, i)
            self.kernels.append(
                ScaleKernel(
                    base_kernel=self.base_kernel, outputscale_prior=per_context_prior
                )
            )

    def forward(self, x1, x2, **params):
        covars = [self.kernels[i](x1=x1, x2=x2) for i in range(self.num_contexts)]
        return reduce(lambda x, y: x + y, covars)


class HierarchicalGP(object):
    def __init__(self, num_contexts, data, lengthscale_prior, outputscale_prior):
        self.kernel = StructureAdditiveKernel(
            num_contexts, lengthscale_prior, outputscale_prior
        )
        self.likelihood = likelihoods.GaussianLikelihood()
        self.data = data

    @bm.random_variable
    def GP(self):
        cov = self.kernel(self.data)
        return gdist.MultivariateNormal(torch.zeros(cov.shape[0]), cov)


class Regression(object):
    def __init__(self, lengthscale_prior):
        self.kernel = ScaleKernel(
            base_kernel=RBFKernel(lengthscale_prior=lengthscale_prior)
        )
        self.mean = gpytorch.means.ZeroMean()
        self.likelihood = likelihoods.GaussianLikelihood()

    # define GP
    @bm.random_variable
    def GP(self, data):
        shape = data.shape[0]
        jitter = torch.eye(shape, shape) * 1e-5
        mean = self.mean(data)
        cov = self.kernel(data) + jitter
        return gdist.MultivariateNormal(mean, cov)


class EvalRegression(gpytorch.models.ExactGP):
    """
    Model to evaluate Regression model
    Note that this is a pure GPytorch model
    """

    def __init__(
        self, train_x, train_y, likelihood, lengthscale=None, lengthscale_prior=None
    ):
        super(EvalRegression, self).__init__(train_x, train_y, likelihood)
        self.kernel = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(lengthscale=lengthscale)
        )
        self.mean = gpytorch.means.ZeroMean()

    def forward(self, data):
        shape = data.shape[0]
        jitter = torch.eye(shape, shape) * 1e-5
        mean = self.mean(data)
        cov = self.kernel(data) + jitter
        return gdist.MultivariateNormal(mean, cov)


class InferenceTests(unittest.TestCase):
    @bm.random_variable
    def lengthscale_prior(self):
        return dist.Gamma(torch.tensor([3.0]), torch.tensor([6.0]))

    @bm.random_variable
    def outputscale_prior(self, i):
        return dist.Gamma(torch.tensor(3.0), torch.tensor(6.0))

    def setUp(self):
        self.mcmc = SingleSiteAncestralMetropolisHastings()
        self.hmc = SingleSiteHamiltonianMonteCarlo(0.05, 10)

    def test_hierarchical_regression_smoke(self):
        x = torch.randn(10, 4)
        y = torch.randn(10)
        num_contexts = 3
        gp = HierarchicalGP(
            num_contexts, x, self.lengthscale_prior, self.outputscale_prior
        )
        obs = {gp.likelihood.forward(gp.GP): y}
        queries = [self.lengthscale_prior()] + [
            self.outputscale_prior(i) for i in range(num_contexts)
        ]
        out = self.mcmc.infer(queries, obs, 1)
        assert len(out.data.rv_dict) == len(queries)

    def test_simple_regression(self):
        torch.manual_seed(1)
        # fit a sin function
        x = torch.linspace(0, 1, 100)
        y = torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * 0.01
        n_samples = 100
        gp = Regression(lengthscale_prior=self.lengthscale_prior)
        gp_prior = partial(gp.GP, x)
        obs = {gp.likelihood.forward(gp_prior): y}
        queries = [self.lengthscale_prior()]
        samples = self.mcmc.infer(queries, obs, n_samples, num_chains=1)
        # get predictives
        predictives = []
        test_x = torch.linspace(0, 1, 51)
        test_y = torch.sin(test_x * (2 * math.pi))
        samples = samples.get_chain(0)[queries[0]]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        for i in range(n_samples):
            pred_gp = EvalRegression(
                x, y, likelihood, lengthscale=torch.tensor(samples[i])
            )
            pred_gp.eval()
            pred_mean = pred_gp.likelihood(pred_gp(test_x)).mean
            predictives.append(pred_mean)
        predictives = torch.stack(predictives, 0)
        pred_y = predictives.mean(0)
        mae = (pred_y - test_y).abs().mean().item()
        assert mae < 0.5
