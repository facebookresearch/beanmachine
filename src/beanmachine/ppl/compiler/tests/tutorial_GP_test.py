# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end test for tutorial on Gaussian Process Regression"""

# This file is a manual replica of the Bento tutorial with the same name
# TODO: The test currently disabled identifies a blocker indicated by the following error message:
# E       RuntimeError: super(): __class__ cell not found
# This looks like a potetion bug in the way we are processing code after reflection

import logging

# TODO: Check imports for conistency
import math
import unittest
from functools import partial

import beanmachine.ppl as bm
import beanmachine.ppl.experimental.gp as bgp
import gpytorch
import matplotlib.pyplot as plt
import torch  # from torch import manual_seed, tensor
import torch.distributions as dist  # from torch.distributions import Bernoulli, Normal, Uniform
from beanmachine.ppl.experimental.gp.models import SimpleGP
from beanmachine.ppl.inference.bmg_inference import BMGInference
from beanmachine.ppl.inference.single_site_no_u_turn_sampler import (
    SingleSiteNoUTurnSampler,
)
from gpytorch.distributions import MultivariateNormal

# This makes the results deterministic and reproducible.

logging.getLogger("beanmachine").setLevel(50)
torch.manual_seed(123)

# Creating sample data

x_train = torch.linspace(0, 1, 11)
y_train = torch.sin(x_train * (2 * math.pi)) + torch.randn(x_train.shape) * 0.2
x_test = torch.linspace(0, 1, 51).unsqueeze(-1)

# MAP Estimation with GPyTorch


class Regression(SimpleGP):
    def __init__(self, x_train, y_train, mean, kernel, likelihood, *args, **kwargs):
        super().__init__(x_train, y_train, mean, kernel, likelihood)

    def forward(self, data, batch_shape=()):
        """
        Computes the GP prior given data. This method should always
        return a `torch.distributions.MultivariateNormal`
        """
        shape = data.shape[len(batch_shape)]
        jitter = torch.eye(shape, shape) * 1e-5
        for _ in range(len(batch_shape)):
            jitter = jitter.unsqueeze(0)
        if isinstance(self.mean, gpytorch.means.Mean):
            # demo using gpytorch for MAP estimation
            mean = self.mean(data)
        else:
            # use Bean Machine for learning posteriors
            if self.training:
                mean = self.mean(batch_shape).expand(data.shape[len(batch_shape) :])
            else:
                mean = self.mean.expand(data.shape[:-1])  # overridden for evaluation
        cov = self.kernel(data) + jitter
        return MultivariateNormal(mean, cov)


kernel = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.PeriodicKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()
mean = gpytorch.means.ConstantMean()
gp = Regression(x_train, y_train, mean, kernel, likelihood)

optimizer = torch.optim.Adam(
    gp.parameters(), lr=0.1
)  # Includes GaussianLikelihood parameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
gp.eval()  # this converts the BM model into a gpytorch model
num_iters = 1

with torch.no_grad():
    observed_pred = likelihood(gp(x_test))
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_train.numpy(), y_train.numpy(), "k*")
    # Plot predictive means as blue line
    ax.plot(x_test.squeeze().numpy(), observed_pred.mean.numpy(), "b")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-1, 1])
    ax.legend(["Observed Data", "Mean", "Confidence"])

# Model


@bm.random_variable
def outputscale():
    return dist.Uniform(torch.tensor(1.0), torch.tensor(2.0))


@bm.random_variable
def lengthscale():
    return dist.Uniform(torch.tensor([0.01]), torch.tensor([0.5]))


@bm.random_variable
def period_length():
    return dist.Uniform(torch.tensor([0.05]), torch.tensor([2.5]))


@bm.random_variable
def noise():
    return dist.Uniform(torch.tensor([0.05]), torch.tensor([0.3]))


@bm.random_variable
def mean(batch_shape=()):
    batch_shape += (1,)
    a = -1 * torch.ones(batch_shape)
    b = torch.ones(batch_shape)
    return dist.Uniform(a, b)


# [Visualization code in tutorial skipped]

# Inference parameters
num_samples = (
    1  ###00 Sample size should not affect (the ability to find) compilation issues.
)

queries = [mean(), lengthscale(), period_length(), outputscale(), noise()]

kernel = bgp.kernels.ScaleKernel(
    base_kernel=bgp.kernels.PeriodicKernel(
        period_length_prior=period_length, lengthscale_prior=lengthscale
    ),
    outputscale_prior=outputscale,
)
likelihood = bgp.likelihoods.GaussianLikelihood(noise_prior=noise)

gp = Regression(x_train, y_train, mean, kernel, likelihood)

gp_prior = partial(gp, x_train)
observations = {gp.likelihood(gp_prior): y_train}


class tutorialGPTest(unittest.TestCase):
    def test_tutorial_GP(self) -> None:
        """Check BM and BMG inference both terminate"""

        self.maxDiff = None

        # Inference with BM

        # Note: No explicit seed here (in original tutorial model). Should we add one?
        nuts = SingleSiteNoUTurnSampler()
        samples = nuts.infer(
            queries,
            observations,
            num_samples=num_samples,
            num_adaptive_samples=10,
            num_chains=1,
        )
        ### The following may be useful for checking correctness of BM/Beanstalk outputs, when there is output
        # lengthscale_samples = samples.get_chain(0)[lengthscale()]
        # outputscale_samples = samples.get_chain(0)[outputscale()]
        # period_length_samples = samples.get_chain(0)[period_length()]
        # mean_samples = samples.get_chain(0)[mean()]
        # noise_samples = samples.get_chain(0)[noise()]

        self.assertTrue(True, msg="We just want to check this point is reached")

    def disabled_test_tutorial_GP_to_dot_cpp_python(
        self,
    ) -> None:
        self.maxDiff = None
        observed = BMGInference().to_dot(queries, observations)
        expected = """
        """
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_cpp(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())

        observed = BMGInference().to_python(queries, observations)
        expected = """
"""
        self.assertEqual(expected.strip(), observed.strip())
